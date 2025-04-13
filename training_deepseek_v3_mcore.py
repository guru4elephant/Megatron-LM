#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import time
import argparse
import numpy as np
from datetime import datetime
from functools import partial

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.optimizer import OptimizerConfig, LRSchedulerConfig, get_optimizer_and_lr_scheduler
from megatron.core.transformer.utils import get_linear_layer
from megatron.core.distributed import initialize_distributed 
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.transformer_layer import ParallelTransformerLayer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

# 添加自定义的initialize_distributed函数
def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, 
                          world_size=None, rank=None, distributed_port=None, 
                          distributed_backend='nccl'):
    """初始化分布式训练环境
    
    Args:
        tensor_model_parallel_size: 张量模型并行大小
        pipeline_model_parallel_size: 流水线模型并行大小
        world_size: 总进程数，如果为None则自动获取
        rank: 当前进程序号，如果为None则自动获取
        distributed_port: 分布式训练端口
        distributed_backend: 分布式后端，默认为'nccl'
    """
    # 清理旧的模型并行状态
    parallel_state.destroy_model_parallel()
    
    # 如果未指定rank和world_size，使用环境变量获取
    if rank is None:
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        elif 'LOCAL_RANK' in os.environ:
            rank = int(os.environ['LOCAL_RANK'])
        else:
            rank = 0
            
    if world_size is None:
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = torch.cuda.device_count()
    
    # 设置当前设备
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # 初始化进程组
    init_method = 'env://'
    if distributed_port is not None:
        init_method = f'tcp://localhost:{distributed_port}'
    
    # 初始化pytorch分布式
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=distributed_backend,
            world_size=world_size,
            rank=rank,
            init_method=init_method
        )
    
    # 初始化Megatron模型并行
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    
    # 打印分布式设置信息
    if torch.distributed.get_rank() == 0:
        print(f"> initialized distributed with backend: {distributed_backend}")
        print(f"> world size: {world_size}")
        print(f"> tensor model parallel size: {tensor_model_parallel_size}")
        print(f"> pipeline model parallel size: {pipeline_model_parallel_size}")

# 直接从modeling_deepseek_v3.py导入关键组件
from modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3MLP,
    DeepseekV3MoE,
    MoEGate,
    DeepseekV3Attention,
    apply_rotary_pos_emb
)

# 为DeepSeek V3实现自定义Transformer层
class DeepseekV3TransformerLayer(MegatronModule):
    """
    DeepSeek V3 Transformer层的自定义实现
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # 规范化层
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 注意力机制
        self.self_attn = self._build_attention()
        
        # MLP或MoE
        self.mlp = self._build_mlp()
    
    def _build_attention(self):
        # 构建注意力机制，适配张量并行
        return DeepseekV3Attention(
            config=self.config,
            layer_idx=self.layer_idx
        )
    
    def _build_mlp(self):
        # 构建MLP或MoE，基于配置选择
        if (
            self.config.n_routed_experts is not None
            and self.layer_idx >= self.config.first_k_dense_replace
            and self.layer_idx % self.config.moe_layer_freq == 0
        ):
            return DeepseekV3MoE(self.config)
        else:
            return DeepseekV3MLP(self.config)
    
    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        position_ids=None,
        past_key_value=None,
        use_cache=False, 
        output_attentions=False
    ):
        # 按照DeepSeek V3的前向传播逻辑
        residual = hidden_states
        
        # 自注意力部分
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        if isinstance(attention_output, tuple):
            hidden_states = attention_output[0]
            presents = attention_output[2] if use_cache else None
            attention_weights = attention_output[1] if output_attentions else None
        else:
            hidden_states = attention_output
            presents = None
            attention_weights = None
        
        hidden_states = residual + hidden_states
        
        # MLP或MoE部分
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attention_weights,)
        
        if use_cache:
            outputs += (presents,)
        
        return outputs


# 重新实现DeepSeek V3模型，完全按照modeling_deepseek_v3.py的逻辑
class DeepseekV3MegatronModel(MegatronModule):
    """
    DeepSeek V3模型的Megatron Core封装，遵循modeling_deepseek_v3.py中的架构
    """
    def __init__(
        self,
        config,
        vocab_size=None,
        max_sequence_length=None,
        pre_process=True,
        post_process=True,
    ):
        super().__init__(share_word_embeddings=True)
        
        # 模型配置
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        
        # 确保vocab_size和max_sequence_length正确设置
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
        if max_sequence_length is not None:
            self.config.max_position_embeddings = max_sequence_length
        
        # 创建模型组件
        if self.pre_process:
            self.embedding = self._build_embedding()
        
        # 创建Transformer层堆叠
        self.layers = self._build_transformer_layers()
        
        if self.post_process:
            self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_layer = self._build_output_layer()
    
    def _build_embedding(self):
        # 词嵌入
        return torch.nn.Embedding(
            self.config.vocab_size, 
            self.config.hidden_size, 
            padding_idx=self.config.pad_token_id
        )
    
    def _build_transformer_layers(self):
        # 构建自定义DeepSeek V3 Transformer层堆叠
        return torch.nn.ModuleList([
            DeepseekV3TransformerLayer(self.config, layer_idx)
            for layer_idx in range(self.config.num_hidden_layers)
        ])
    
    def _build_output_layer(self):
        # 语言模型头
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            return RowParallelLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=self.config.initializer_range),
            )
        else:
            return torch.nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False
            )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 按照DeepSeek V3的前向传播逻辑实现
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        
        # 词嵌入
        if inputs_embeds is None:
            if self.pre_process:
                inputs_embeds = self.embedding(input_ids)
            else:
                raise ValueError("如果pre_process=False，必须提供inputs_embeds")
        
        # 初始化hidden_states
        hidden_states = inputs_embeds
        
        # 注意力掩码处理
        if attention_mask is not None and attention_mask.dim() == 2:
            # 确保注意力掩码格式正确 [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将0/1掩码转换为-inf/0掩码
            attention_mask = (1.0 - attention_mask.to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
        
        # 准备past_key_values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # 存储中间状态的列表
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        next_decoder_cache = [] if use_cache else None
        
        # 通过Transformer层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache.append(layer_outputs[-1])
            
            if output_attentions:
                all_self_attns.append(layer_outputs[1])
        
        # 最终规范化和语言模型头
        if self.post_process:
            # 最终层规范化
            hidden_states = self.norm(hidden_states)
            
            # 输出投影
            logits = self.output_layer(hidden_states)
            
            # 如果使用张量并行，收集完整的logits
            if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.config.no_gather_logits is not True:
                logits = gather_from_tensor_model_parallel_region(logits)
        else:
            logits = None
        
        # 计算损失
        loss = None
        if labels is not None and logits is not None:
            # 计算因果语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 在使用模型并行时启用
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        # 准备输出
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_self_attns,)
            return ((loss,) + output) if loss is not None else output
        
        # 返回损失和logits (简化输出以匹配之前的接口)
        return (loss, logits) if loss is not None else (logits,)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepSeek V3 with Megatron Core Training')
    
    # 模型参数
    parser.add_argument('--num-layers', type=int, default=32, help='模型层数')
    parser.add_argument('--hidden-size', type=int, default=4096, help='隐藏层大小')
    parser.add_argument('--num-attention-heads', type=int, default=32, help='注意力头数')
    parser.add_argument('--num-key-value-heads', type=int, default=32, help='键值注意力头数量')
    parser.add_argument('--intermediate-size', type=int, default=11008, help='中间层大小')
    parser.add_argument('--max-position-embeddings', type=int, default=4096, help='最大位置编码长度')
    parser.add_argument('--vocab-size', type=int, default=131072, help='词表大小')
    parser.add_argument('--use-moe', action='store_true', help='是否使用MoE')
    parser.add_argument('--num-experts', type=int, default=8, help='MoE专家数量')
    parser.add_argument('--experts-per-token', type=int, default=2, help='每个token使用的专家数量')
    parser.add_argument('--moe-layer-freq', type=int, default=2, help='MoE层的频率')
    
    # 训练参数
    parser.add_argument('--train-iters', type=int, default=1000000, help='训练迭代次数')
    parser.add_argument('--batch-size', type=int, default=16, help='全局批处理大小')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='梯度累积步数')
    parser.add_argument('--lr', type=float, default=1.0e-4, help='学习率')
    parser.add_argument('--min-lr', type=float, default=1.0e-5, help='最小学习率')
    parser.add_argument('--lr-decay-style', type=str, default='cosine', help='学习率衰减策略')
    parser.add_argument('--lr-warmup-iters', type=int, default=2000, help='学习率预热迭代次数')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--bf16', action='store_true', help='使用BF16精度')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True, help='数据集路径')
    parser.add_argument('--max-sequence-length', type=int, default=2048, help='最大序列长度')
    
    # 分布式训练参数
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='张量模型并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='流水线模型并行大小')
    parser.add_argument('--expert-model-parallel-size', type=int, default=1, help='专家模型并行大小')
    parser.add_argument('--distributed-backend', default='nccl', choices=['nccl', 'gloo'], help='分布式后端')
    
    # 日志和检查点参数
    parser.add_argument('--save', type=str, default='checkpoints', help='保存检查点的目录')
    parser.add_argument('--load', type=str, default=None, help='加载检查点的路径')
    parser.add_argument('--log-interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--save-interval', type=int, default=1000, help='保存检查点间隔')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--local_rank', type=int, default=None, help='本地进程排名(由torch.distributed自动提供)')
    
    return parser.parse_args()


def setup_deepseek_v3_config(args):
    """设置DeepSeek V3配置"""
    from transformers import PretrainedConfig
    
    config = PretrainedConfig()
    
    # 核心模型参数
    config.num_hidden_layers = args.num_layers
    config.hidden_size = args.hidden_size
    config.num_attention_heads = args.num_attention_heads
    config.num_key_value_heads = args.num_key_value_heads
    config.intermediate_size = args.intermediate_size
    config.hidden_act = "silu"
    config.max_position_embeddings = args.max_position_embeddings
    config.rms_norm_eps = 1e-6
    config.pad_token_id = 0
    config.initializer_range = 0.02
    config.attention_bias = False
    config.rope_theta = 10000.0
    config.no_gather_logits = False
    
    # MoE相关参数
    if args.use_moe:
        config.n_routed_experts = args.num_experts
        config.num_experts_per_tok = args.experts_per_token
        config.moe_layer_freq = args.moe_layer_freq
        config.first_k_dense_replace = 0  # 从哪一层开始替换为MoE
        config.moe_intermediate_size = args.intermediate_size
        config.n_shared_experts = None
        config.router_aux_loss_coef = 0.01
        config.topk_method = "noaux_tc"
        config.n_group = 1
        config.topk_group = 1
        config.moe_scoring_func = "sigmoid"
    else:
        config.n_routed_experts = None
    
    # 训练参数
    config.fp16 = args.fp16
    config.bf16 = args.bf16
    
    # 其他参数
    config._attn_implementation = "eager"  # 在Megatron中不使用Flash Attention 2
    
    return config


def prepare_model_and_optimizer(args):
    """准备模型和优化器"""
    config = setup_deepseek_v3_config(args)
    
    # 初始化模型
    model = DeepseekV3MegatronModel(
        config=config,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_sequence_length,
        pre_process=True,
        post_process=True
    )
    
    # 配置优化器
    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        use_cpu_initialization=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad
    )
    
    # 配置学习率调度器
    lr_scheduler_config = LRSchedulerConfig(
        lr_decay_style=args.lr_decay_style,
        warmup_steps=args.lr_warmup_iters,
        min_lr=args.min_lr,
        total_steps=args.train_iters,
    )
    
    # 创建优化器和调度器
    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
        model=model,
        config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    return model, optimizer, lr_scheduler


def load_data(args):
    """加载和准备数据集"""
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
    
    # 配置GPT数据集
    data_config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.max_sequence_length,
        split="969, 30, 1",
        path_to_corpus=args.data_path,
        tokenizer_type="DeepSeekV3Tokenizer",
    )
    
    # 创建数据集构建器
    data_builder = BlendedMegatronDatasetBuilder(
        data_config,
        args.batch_size,
        args.gradient_accumulation_steps,
        drop_last=True,
    )
    
    # 构建数据集和数据加载器
    train_dataset, val_dataset = data_builder.build_train_valid_test_datasets(
        train_valid_test_ratio=[0.98, 0.02, 0.0]
    )
    
    train_dataloader = data_builder.build_train_valid_test_data_loaders(
        train_dataset, val_dataset, None
    )[0]
    
    return train_dataloader


def train(args, model, optimizer, lr_scheduler, train_dataloader):
    """训练循环"""
    from megatron.core.gradient_accumulation_fusion import get_grad_accumulation_object
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    
    model.train()
    
    # 获取前向后向函数
    fwd_bwd_function = get_forward_backward_func(
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        grad_scaler=None
    )
    
    # 梯度累积实现
    grad_accum_object = get_grad_accumulation_object(args.gradient_accumulation_steps)
    
    total_loss = 0.0
    step = 0
    start_time = time.time()
    
    # 创建dataloader迭代器
    train_dataloader_iter = iter(train_dataloader)
    
    # 训练循环
    for iteration in range(args.train_iters):
        optimizer.zero_grad()
        
        # 累积梯度循环
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_dataloader_iter)
            except (StopIteration, AttributeError):
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)
            
            # 解包批次数据
            tokens = batch["text"]
            labels = batch["labels"] if "labels" in batch else tokens
            attention_mask = (tokens != args.pad_token_id).long()
            
            # 计算位置ID
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            
            # 前向和后向传播
            losses = fwd_bwd_function(
                forward_step_func=partial(
                    forward_step,
                    model=model,
                    tokens=tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels
                ),
                data_iterator=None,
                model=model,
                num_microbatches=1,
                seq_length=args.max_sequence_length,
                micro_batch_size=tokens.size(0),
                grad_scaler=None,
            )
            
            # 平均损失
            loss = losses[0]
            total_loss += loss.item()
            
            # 梯度累积
            grad_accum_object.register_forward_step(0)
            grad_accum_object.register_backward_step()
            
            if grad_accum_object.is_gradient_accumulation_boundary():
                # 应用梯度裁剪并更新参数
                optimizer.step()
                lr_scheduler.step()
        
        step += 1
        
        # 记录日志
        if step % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed_time = time.time() - start_time
            tokens_per_sec = (args.batch_size * args.max_sequence_length * args.log_interval) / elapsed_time
            
            if parallel_state.get_data_parallel_rank() == 0:
                print(f"Iteration: {step} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_lr():.8f} | "
                      f"Tokens/s: {tokens_per_sec:.2f}")
            
            total_loss = 0.0
            start_time = time.time()
        
        # 保存检查点
        if step % args.save_interval == 0:
            save_checkpoint(args, model, optimizer, lr_scheduler, step)


def forward_step(model, tokens, attention_mask, position_ids, labels):
    """前向步骤函数"""
    outputs = model(
        input_ids=tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels
    )
    
    return outputs[0]  # 返回损失


def save_checkpoint(args, model, optimizer, lr_scheduler, iteration):
    """保存模型检查点"""
    from megatron.core.transformer.utils import make_sure_torch_distributed_is_initialized
    from megatron.core.transformer.checkpointing import add_checkpoint_callback_arguments, save_checkpoint, load_checkpoint
    
    make_sure_torch_distributed_is_initialized()
    
    if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
        print(f"保存检查点到 {args.save}/iter_{iteration}")
    
    # 创建检查点目录
    os.makedirs(f"{args.save}/iter_{iteration}", exist_ok=True)
    
    checkpoint_args = {
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    
    # 保存检查点
    save_checkpoint(model, f"{args.save}/iter_{iteration}", checkpoint_args)


def load_checkpoint_if_exists(args, model, optimizer, lr_scheduler):
    """如果存在检查点，则加载"""
    from megatron.core.transformer.checkpointing import load_checkpoint
    
    if args.load is None:
        return 0  # 从头开始训练
    
    print(f"尝试从 {args.load} 加载检查点")
    iteration = load_checkpoint(model, args.load, optimizer, lr_scheduler)
    print(f"成功加载检查点，从迭代 {iteration} 开始继续训练")
    
    return iteration


def main():
    """主函数"""
    # 解析参数
    args = get_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化分布式环境
    initialize_distributed(
        tensor_model_parallel_size=args.tensor_model_parallel_size, 
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        world_size=None,
        rank=None,
        distributed_port=None,
        distributed_backend=args.distributed_backend
    )
    
    # 设置CUDA随机种子
    model_parallel_cuda_manual_seed(args.seed)
    
    # 准备模型和优化器
    model, optimizer, lr_scheduler = prepare_model_and_optimizer(args)
    
    # 如果有检查点，加载
    start_iteration = load_checkpoint_if_exists(args, model, optimizer, lr_scheduler)
    
    # 加载数据
    train_dataloader = load_data(args)
    
    # 训练模型
    train(args, model, optimizer, lr_scheduler, train_dataloader)
    
    # 保存最终模型
    save_checkpoint(args, model, optimizer, lr_scheduler, args.train_iters)
    
    # 清理
    parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    main() 