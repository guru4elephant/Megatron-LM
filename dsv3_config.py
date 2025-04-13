#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSeekV3模型配置")
    
    # 模型参数
    parser.add_argument('--hidden-size', type=int, default=1024, 
                        help='隐藏状态大小')
    parser.add_argument('--num-layers', type=int, default=32, 
                        help='Transformer层数量')
    parser.add_argument('--num-attention-heads', type=int, default=32, 
                        help='注意力头的数量')
    parser.add_argument('--kv-channels', type=int, default=128, 
                        help='Key和Value通道数')
    parser.add_argument('--ffn-hidden-size', type=int, default=2816, 
                        help='前馈网络隐藏大小')
    parser.add_argument('--rotary-percent', type=float, default=1.0, 
                        help='旋转嵌入百分比')
    
    # MoE参数
    parser.add_argument('--num-moe-experts', type=int, default=0, 
                        help='MoE专家数量，0表示不使用MoE')
    parser.add_argument('--moe-frequency', type=int, default=0, 
                        help='MoE层频率，0表示不使用MoE')
    parser.add_argument('--moe-expert-parallel-size', type=int, default=1, 
                        help='MoE专家并行大小')
    parser.add_argument('--moe-router-topk', type=int, default=2, 
                        help='MoE路由器选择的专家数量')
    
    # 词汇表参数
    parser.add_argument('--vocab-size', type=int, default=100352, 
                        help='词汇表大小')
    parser.add_argument('--max-position-embeddings', type=int, default=8192, 
                        help='位置嵌入最大长度')
    
    # 训练参数
    parser.add_argument('--micro-batch-size', type=int, default=1, 
                        help='单GPU批次大小')
    parser.add_argument('--global-batch-size', type=int, default=32, 
                        help='全局批次大小')
    parser.add_argument('--learning-rate', type=float, default=6e-5, 
                        help='学习率')
    parser.add_argument('--min-learning-rate', type=float, default=6e-6, 
                        help='最小学习率')
    parser.add_argument('--weight-decay', type=float, default=0.1, 
                        help='权重衰减')
    parser.add_argument('--adam-beta1', type=float, default=0.9, 
                        help='Adam优化器beta1')
    parser.add_argument('--adam-beta2', type=float, default=0.95, 
                        help='Adam优化器beta2')
    parser.add_argument('--clip-grad', type=float, default=1.0, 
                        help='梯度裁剪')
    
    # 并行处理参数
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, 
                        help='模型并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, 
                        help='管道并行大小')
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--fp16', action='store_true',
                        help='是否使用FP16精度')
    parser.add_argument('--bf16', action='store_true', 
                        help='是否使用BF16精度')
    parser.add_argument('--checkpoint-activations', action='store_true',
                        help='是否检查点激活')
    parser.add_argument('--checkpoint-num-layers', type=int, default=1,
                        help='每隔多少层检查点激活')
    
    # 训练数据参数
    parser.add_argument('--data-path', type=str, default=None, 
                        help='训练数据路径')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='预训练模型路径')
    
    return parser.parse_args()


@dataclass
class DeepSeekV3Config:
    """DeepSeekV3配置类"""
    
    # 模型架构参数
    hidden_size: int = 1024
    num_layers: int = 32
    num_attention_heads: int = 32
    kv_channels: int = 128
    ffn_hidden_size: int = 2816
    rotary_percent: float = 1.0
    layernorm_epsilon: float = 1e-5
    vocab_size: int = 100352
    max_position_embeddings: int = 8192
    
    # 正则化与优化参数
    hidden_dropout: float = 0.0  # DeepSeekV3使用0.0的dropout率
    attention_dropout: float = 0.0  # DeepSeekV3使用0.0的注意力dropout率
    
    # 激活函数
    activation_func: str = "swiglu"  # DeepSeekV3使用SwiGLU激活函数
    
    # MoE参数
    num_moe_experts: int = 0  # 默认不使用MoE
    moe_frequency: int = 0  # 默认不使用MoE
    moe_router_topk: int = 2
    
    # 初始化参数
    init_method_std: float = 0.02
    use_scaled_init_method: bool = True
    
    # 位置编码参数
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000
    
    # 其他参数
    apply_residual_connection_post_layernorm: bool = False
    add_bias_linear: bool = False  # DeepSeekV3线性层不使用偏置
    openai_gelu: bool = False
    
    # 训练参数
    gradient_accumulation_steps: int = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        args = get_args()
        
        # 从命令行更新参数
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_attention_heads = args.num_attention_heads
        self.kv_channels = args.kv_channels
        self.ffn_hidden_size = args.ffn_hidden_size
        self.rotary_percent = args.rotary_percent
        self.vocab_size = args.vocab_size
        self.max_position_embeddings = args.max_position_embeddings
        
        # MoE参数
        self.num_moe_experts = args.num_moe_experts
        self.moe_frequency = args.moe_frequency
        
        # 计算梯度累积步数
        if hasattr(args, "global_batch_size") and hasattr(args, "micro_batch_size"):
            micro_batch_size = args.micro_batch_size
            global_batch_size = args.global_batch_size
            data_parallel_size = 1
            
            if hasattr(args, "tensor_model_parallel_size"):
                data_parallel_size //= args.tensor_model_parallel_size
                
            if hasattr(args, "pipeline_model_parallel_size"):
                data_parallel_size //= args.pipeline_model_parallel_size
                
            if data_parallel_size <= 0:
                data_parallel_size = 1
                
            self.gradient_accumulation_steps = max(
                global_batch_size // (micro_batch_size * data_parallel_size), 1
            )
    
    def create_transformer_config(self) -> TransformerConfig:
        """创建用于Megatron Core的TransformerConfig实例"""
        args = get_args()
        
        # 设置精度类型
        if args.fp16:
            precision = torch.float16
        elif args.bf16:
            precision = torch.bfloat16
        else:
            precision = torch.float32
        
        # 将DeepSeekV3Config参数转化为TransformerConfig参数
        config_dict = {
            # 从DeepSeekV3Config复制的参数
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "kv_channels": self.kv_channels,
            "ffn_hidden_size": self.ffn_hidden_size,
            "layernorm_epsilon": self.layernorm_epsilon,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "activation_func": self.activation_func,
            "init_method_std": self.init_method_std,
            "apply_residual_connection_post_layernorm": self.apply_residual_connection_post_layernorm,
            "add_bias_linear": self.add_bias_linear,
            "openai_gelu": self.openai_gelu,
            
            # 特殊参数
            "use_rotary_position_embeddings": self.use_rotary_position_embeddings,
            "rotary_percent": self.rotary_percent,
            "rotary_base": self.rotary_base,
            
            # MoE参数
            "num_moe_experts": self.num_moe_experts,
            "moe_frequency": self.moe_frequency,
            "moe_router_topk": self.moe_router_topk,
            
            # 从args获取的参数
            "fp16": args.fp16,
            "bf16": args.bf16,
            "precision": precision,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": args.seed,
            "params_dtype": precision,
            
            # 添加DeepSeekV3特定配置
            "tie_word_embeddings": False,
            "use_flash_attn": True,  # 默认使用Flash Attention
            "normalization": "rmsnorm",  # 使用RMSNorm
        }
        
        return TransformerConfig(**config_dict)


if __name__ == "__main__":
    # 测试配置
    config = DeepSeekV3Config()
    transformer_config = config.create_transformer_config()
    
    print("DeepSeekV3配置:")
    print(f"隐藏大小: {config.hidden_size}")
    print(f"层数: {config.num_layers}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"KV通道数: {config.kv_channels}")
    print(f"FFN隐藏大小: {config.ffn_hidden_size}")
    print(f"词汇表大小: {config.vocab_size}")
    
    print("\nTransformerConfig:")
    print(f"隐藏大小: {transformer_config.hidden_size}")
    print(f"层数: {transformer_config.num_layers}")
    print(f"注意力头数: {transformer_config.num_attention_heads}")
    print(f"梯度累积步数: {transformer_config.gradient_accumulation_steps}") 