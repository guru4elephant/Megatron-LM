#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Megatron Core 训练示例脚本
此脚本演示如何使用Megatron Core创建一个简单的模型并进行训练
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import datetime
import json
import threading
import wandb
from collections import defaultdict

# Megatron Core 导入
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed import initialize_distributed
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

try:
    from megatron.core.timers import Timers
except ImportError:
    # 简单的Timers实现，以防megatron中的不可用
    class Timers:
        def __init__(self):
            self.timers = {}
            
        def __call__(self, name):
            if name not in self.timers:
                self.timers[name] = Timer(name)
            return self.timers[name]
            
        def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False):
            if torch.distributed.get_rank() == 0:
                print('Timers:')
                for name in names:
                    if name in self.timers:
                        timer = self.timers[name]
                        elapsed_time = timer.elapsed(reset=reset) / normalizer
                        print(f'    {name}: {elapsed_time:.2f} ms')
                        
class Timer:
    """计时器类，用于测量不同操作的耗时"""
    def __init__(self, name):
        self.name = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = 0.0
        
    def start(self):
        """开始计时"""
        assert not self.started_, f"计时器 {self.name} 已经启动"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True
        
    def stop(self):
        """停止计时"""
        assert self.started_, f"计时器 {self.name} 未启动"
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time) * 1000  # 转换为毫秒
        self.started_ = False
        
    def reset(self):
        """重置计时器"""
        self.elapsed_ = 0.0
        self.started_ = False
        
    def elapsed(self, reset=True):
        """返回经过的时间"""
        elapsed = self.elapsed_
        if reset:
            self.reset()
        return elapsed

class MemoryTracker:
    """用于跟踪GPU内存使用情况的类"""
    def __init__(self, enabled=True, log_interval=10):
        self.enabled = enabled
        self.log_interval = log_interval
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.stats_history = []
        self.continuous_monitoring = False
        self.monitor_thread = None
        
    def start_continuous_monitoring(self, interval=1.0):
        """开始持续监控内存使用情况"""
        if not self.enabled:
            return
            
        self.continuous_monitoring = True
        
        def monitor_func():
            while self.continuous_monitoring:
                self.capture_stats("continuous")
                time.sleep(interval)
                
        self.monitor_thread = threading.Thread(target=monitor_func)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_continuous_monitoring(self):
        """停止持续监控"""
        self.continuous_monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
            
    def capture_stats(self, label=""):
        """捕获当前内存使用情况"""
        if not self.enabled:
            return {}
            
        # 确保在测量前完成所有CUDA操作
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        
        # 更新峰值
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        
        stats = {
            "timestamp": time.time(),
            "label": label,
            "allocated_GB": allocated,
            "reserved_GB": reserved,
            "peak_allocated_GB": self.peak_allocated,
            "peak_reserved_GB": self.peak_reserved,
        }
        
        self.stats_history.append(stats)
        return stats
        
    def log_stats(self, label=""):
        """记录内存使用情况"""
        if not self.enabled:
            return
            
        stats = self.capture_stats(label)
        
        if torch.distributed.get_rank() == 0:
            print(f"\n=== Memory Stats {label} ===")
            print(f"Allocated: {stats['allocated_GB']:.2f} GB")
            print(f"Peak Allocated: {stats['peak_allocated_GB']:.2f} GB")
            print(f"Reserved: {stats['reserved_GB']:.2f} GB")
            print(f"Peak Reserved: {stats['peak_reserved_GB']:.2f} GB")
            
        return stats
        
    def get_memory_summary(self):
        """返回内存使用摘要"""
        if not self.enabled:
            return {}
            
        return {
            "peak_allocated_GB": self.peak_allocated,
            "peak_reserved_GB": self.peak_reserved,
            "current_allocated_GB": torch.cuda.memory_allocated() / (1024 ** 3),
            "current_reserved_GB": torch.cuda.memory_reserved() / (1024 ** 3),
        }
        
    def reset_peaks(self):
        """重置峰值内存记录"""
        if not self.enabled:
            return
            
        self.peak_allocated = 0
        self.peak_reserved = 0

class PerformanceTracker:
    """跟踪和记录模型性能指标的类"""
    def __init__(self, enabled=True, wandb_enabled=False):
        self.enabled = enabled
        self.wandb_enabled = wandb_enabled
        self.timers = Timers()
        self.memory_tracker = MemoryTracker(enabled=enabled)
        self.global_step = 0
        self.metrics = defaultdict(list)
        
    def start_timer(self, name):
        """开始指定名称的计时器"""
        if not self.enabled:
            return
        self.timers(name).start()
        
    def stop_timer(self, name):
        """停止指定名称的计时器"""
        if not self.enabled:
            return
        self.timers(name).stop()
        
    def log_memory(self, label=""):
        """记录内存使用情况"""
        if not self.enabled:
            return {}
        return self.memory_tracker.log_stats(label)
        
    def log_iteration_metrics(self, metrics_dict, step=None):
        """记录每个训练迭代的指标"""
        if not self.enabled:
            return
            
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1
            
        # 记录这一步的性能指标
        for name in ['forward', 'backward', 'optimizer', 'iteration']:
            if name in self.timers.timers:
                elapsed = self.timers(name).elapsed(reset=True)
                metrics_dict[f"{name}_time_ms"] = elapsed
                
        # 获取并添加内存指标
        memory_stats = self.memory_tracker.get_memory_summary()
        metrics_dict.update(memory_stats)
        
        # 存储所有指标的历史记录
        for k, v in metrics_dict.items():
            self.metrics[k].append(v)
            
        # 如果启用了wandb，则发送到wandb
        if self.wandb_enabled and torch.distributed.get_rank() == 0:
            wandb.log(metrics_dict, step=self.global_step)
            
        return metrics_dict
        
    def start_continuous_memory_monitoring(self, interval=1.0):
        """开始持续监控内存使用情况"""
        if not self.enabled:
            return
        self.memory_tracker.start_continuous_monitoring(interval)
        
    def stop_continuous_memory_monitoring(self):
        """停止持续监控内存"""
        if not self.enabled:
            return
        self.memory_tracker.stop_continuous_monitoring()
        
    def get_summary(self):
        """获取所有性能指标的摘要统计信息"""
        if not self.enabled:
            return {}
            
        summary = {}
        
        # 计算每个指标的平均值、最小值、最大值
        for k, v in self.metrics.items():
            if len(v) > 0:
                summary[f"{k}_avg"] = sum(v) / len(v)
                summary[f"{k}_min"] = min(v)
                summary[f"{k}_max"] = max(v)
                
        # 添加内存摘要
        summary.update(self.memory_tracker.get_memory_summary())
        
        return summary
        
    def reset(self):
        """重置所有计时器和指标"""
        if not self.enabled:
            return
        for name in self.timers.timers:
            self.timers(name).reset()
        self.memory_tracker.reset_peaks()
        self.metrics = defaultdict(list)

# 创建一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, vocab_size, seq_length, size=1000):
        self.data = torch.randint(0, vocab_size, (size, seq_length), dtype=torch.long)
        self.size = size
        self.seq_length = seq_length
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 返回输入和标签(下一个token预测)
        x = self.data[idx]
        # 假设标签是预测序列中的下一个token
        y = torch.cat([x[1:], torch.tensor([0])], dim=0)
        return {'text': x, 'labels': y}

def collate_fn(batch):
    input_ids = torch.stack([item['text'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # 创建注意力掩码 (causal mask)
    seq_length = input_ids.size(1)
    attention_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool)).unsqueeze(0).expand(len(batch), -1, -1)
    
    # 创建位置编码
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }

# 自定义模型，继承MegatronModule
class SimpleTransformerModel(MegatronModule):
    def __init__(self, config, transformer_layer_spec, vocab_size, max_seq_length):
        super().__init__(config=config)
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # 创建embedding层
        self.embedding = torch.nn.Embedding(vocab_size, config.hidden_size)
        
        # 创建Transformer Block
        self.transformer = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True
        )
        
        # 输出层
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=True
        )
        
        # 损失函数
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    def set_input_tensor(self, input_tensor):
        """将输入张量设置到TransformerBlock"""
        self.transformer.set_input_tensor(input_tensor)
    
    def forward(self, input_ids, position_ids, attention_mask, labels=None):
        # 对输入进行编码
        embeddings = self.embedding(input_ids)
        
        # 传递给transformer
        hidden_states = self.transformer(embeddings, attention_mask)
        
        # 输出层
        logits, _ = self.output_layer(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 变形logits以适应交叉熵损失 [batch, seq_len, vocab] -> [batch*seq_len, vocab]
            logits_2d = logits.view(-1, logits.size(-1))
            labels_1d = labels.view(-1)
            loss = self.loss_fn(logits_2d, labels_1d)
        
        return loss, logits

def setup_model_and_optimizer(args):
    """设置模型和优化器"""
    
    # 初始化 transformer 配置
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        layernorm_epsilon=1e-5,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        init_method_std=0.02,
        use_cpu_initialization=False,
        fp16=args.fp16,
        bf16=args.bf16,
        apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
        attention_softmax_in_fp32=args.attention_softmax_in_fp32,
        kv_channels=args.hidden_size // args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
    )
    
    # 创建transformer layer规范
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(config)
    
    # 创建模型
    model = SimpleTransformerModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.vocab_size,
        max_seq_length=args.seq_length
    )
    
    # 如果启用了混合精度训练
    if args.fp16 or args.bf16:
        if args.fp16:
            model.half()
        else:
            model.bfloat16()
    
    # 将模型移动到GPU
    model.cuda(torch.cuda.current_device())
    
    # 配置优化器
    optimizer_config = OptimizerConfig(
        optimizer_type='adam',
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    
    # 创建优化器
    optimizer = get_megatron_optimizer(optimizer_config, model)
    
    # 创建学习率调度器
    lr_scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.train_steps,
        lr_decay_style='linear',
    )
    
    return model, optimizer, lr_scheduler

def train(args):
    """训练函数"""
    
    # 初始化分布式环境
    initialize_distributed(
        init_method=args.init_method,
        ranks_filename=None,
        rank=args.local_rank,
        world_size=args.world_size,
    )
    
    # 初始化模型并行
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    )
    
    # 初始化性能跟踪器
    tracker = PerformanceTracker(enabled=True, wandb_enabled=args.wandb)
    
    # 如果使用wandb且是主进程，则初始化wandb
    if args.wandb and torch.distributed.get_rank() == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"megatron-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel_size}",
            config=vars(args),
            group=args.wandb_group,
        )
    
    # 打印一些运行信息
    if torch.distributed.get_rank() == 0:
        print(f"世界大小: {torch.distributed.get_world_size()}")
        print(f"数据并行度: {get_data_parallel_world_size()}")
        print(f"Tensor并行度: {get_tensor_model_parallel_world_size()}")
        print(f"Pipeline并行度: {get_pipeline_model_parallel_world_size()}")
        print(f"Sequence并行: {args.sequence_parallel}")
        
        # 记录GPU信息
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"CUDA版本: {torch.version.cuda}")
    
    # 开始持续监控内存（根据需要）
    if args.monitor_memory_continuously:
        tracker.start_continuous_memory_monitoring(interval=args.memory_monitor_interval)
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        size=args.dataset_size
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=get_data_parallel_world_size(),
        rank=get_data_parallel_rank(),
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # 记录初始内存状态
    tracker.log_memory("初始化前")
    
    # 创建模型和优化器
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    
    # 记录模型加载后的内存状态
    tracker.log_memory("模型加载后")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    if torch.distributed.get_rank() == 0:
        print(f"模型总参数量: {total_params / 1000000:.2f}M")
        
        if args.wandb:
            wandb.log({"model_parameters_M": total_params / 1000000})
    
    # 训练循环
    step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # 准备输入
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            position_ids = batch['position_ids'].cuda()
            
            # 记录当前迭代开始
            tracker.start_timer('iteration')
            
            # 前向传播
            tracker.start_timer('forward')
            tracker.log_memory("前向传播前")
            
            loss, _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            tracker.stop_timer('forward')
            tracker.log_memory("前向传播后")
            
            # 反向传播
            tracker.start_timer('backward')
            
            optimizer.zero_grad()
            loss.backward()
            
            tracker.stop_timer('backward')
            tracker.log_memory("反向传播后")
            
            # 梯度裁剪(可选)
            if args.gradient_clip_val > 0:
                tracker.start_timer('gradient_clip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                tracker.stop_timer('gradient_clip')
            
            # 优化器和学习率调度器步骤
            tracker.start_timer('optimizer')
            
            optimizer.step()
            lr_scheduler.step()
            
            tracker.stop_timer('optimizer')
            tracker.log_memory("优化器步骤后")
            
            # 记录本次迭代结束
            tracker.stop_timer('iteration')
            
            # 计算每秒处理的样本数和每秒处理的token数
            iter_time = tracker.timers('iteration').elapsed() / 1000.0  # 转换为秒
            samples_per_sec = args.batch_size / iter_time
            tokens_per_sec = args.batch_size * args.seq_length / iter_time
            
            # 记录指标
            metrics = {
                'loss': loss.item(),
                'learning_rate': lr_scheduler.get_lr(),
                'epoch': epoch,
                'global_step': step,
                'samples_per_sec': samples_per_sec,
                'tokens_per_sec': tokens_per_sec,
            }
            
            # 打印日志
            if get_data_parallel_rank() == 0 and step % args.log_interval == 0:
                time_elapsed = time.time() - epoch_start_time
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, "
                      f"LR: {lr_scheduler.get_lr():.6f}, "
                      f"Samples/sec: {samples_per_sec:.2f}, "
                      f"Tokens/sec: {tokens_per_sec:.2f}, "
                      f"Time: {time_elapsed:.2f}s")
                
                # 记录计时器状态
                tracker.timers.log(['forward', 'backward', 'optimizer', 'iteration'])
            
            # 记录到性能跟踪器
            tracker.log_iteration_metrics(metrics, step)
            
            step += 1
            
            # 检查是否达到最大训练步数
            if step >= args.train_steps:
                break
        
        # 每轮结束后打印摘要
        if get_data_parallel_rank() == 0:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} 完成，用时: {epoch_time:.2f}秒")
        
        # 检查是否达到最大训练步数
        if step >= args.train_steps:
            break
    
    # 停止内存监控
    if args.monitor_memory_continuously:
        tracker.stop_continuous_memory_monitoring()
    
    # 记录结束时的内存状态
    tracker.log_memory("训练结束")
    
    # 获取并打印性能摘要
    if get_data_parallel_rank() == 0:
        summary = tracker.get_summary()
        print("\n===== 训练性能摘要 =====")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")
        
        # 将摘要保存到文件
        if args.save_performance_summary:
            with open(args.performance_summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
        # 记录到wandb
        if args.wandb:
            wandb.log({"summary": summary})
            wandb.finish()
    
    print("训练完成!")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="Megatron Core训练示例")
    
    # 模型参数
    parser.add_argument('--num-layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--hidden-size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--num-attention-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--ffn-hidden-size', type=int, default=1024, help='前馈网络隐藏层大小')
    parser.add_argument('--vocab-size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--seq-length', type=int, default=256, help='序列长度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--train-steps', type=int, default=1000, help='训练步数')
    parser.add_argument('--dataset-size', type=int, default=10000, help='数据集大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--adam-beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam-beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--adam-eps', type=float, default=1e-08, help='Adam epsilon')
    parser.add_argument('--warmup-steps', type=int, default=100, help='预热步数')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0, help='梯度裁剪值')
    parser.add_argument('--attention-dropout', type=float, default=0.1, help='注意力dropout')
    parser.add_argument('--hidden-dropout', type=float, default=0.1, help='隐藏层dropout')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载器工作进程数')
    
    # 并行参数
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='Tensor并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='Pipeline并行大小')
    parser.add_argument('--sequence-parallel', action='store_true', help='启用序列并行')
    
    # 分布式参数
    parser.add_argument('--local-rank', type=int, default=0, help='局部rank')
    parser.add_argument('--world-size', type=int, default=1, help='世界大小')
    parser.add_argument('--init-method', type=str, default='tcp://localhost:12345', help='初始化方法')
    
    # 精度参数
    parser.add_argument('--fp16', action='store_true', help='使用FP16')
    parser.add_argument('--bf16', action='store_true', help='使用BF16')
    parser.add_argument('--apply-query-key-layer-scaling', action='store_true', help='应用query-key缩放')
    parser.add_argument('--attention-softmax-in-fp32', action='store_true', help='在FP32中计算softmax')
    
    # 性能监控参数
    parser.add_argument('--monitor-memory-continuously', action='store_true', help='持续监控内存使用情况')
    parser.add_argument('--memory-monitor-interval', type=float, default=1.0, help='内存监控间隔(秒)')
    parser.add_argument('--save-performance-summary', action='store_true', help='保存性能摘要到文件')
    parser.add_argument('--performance-summary-file', type=str, default='performance_summary.json', help='性能摘要文件路径')
    
    # Weights & Biases相关参数
    parser.add_argument('--wandb', action='store_true', help='启用Weights & Biases记录')
    parser.add_argument('--wandb-project', type=str, default='megatron-core', help='wandb项目名称')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb运行名称')
    parser.add_argument('--wandb-group', type=str, default=None, help='wandb分组名称')
    
    args = parser.parse_args()
    
    # 确保FP16和BF16不会同时启用
    if args.fp16 and args.bf16:
        raise ValueError("FP16和BF16不能同时启用")
    
    # 默认attention_softmax_in_fp32为True
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True
    
    train(args)

if __name__ == "__main__":
    main() 