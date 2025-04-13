#!/usr/bin/env python
# coding=utf-8

import os
import torch
import argparse
import numpy as np
from functools import partial

# 导入Megatron Core的必要组件
from megatron.core import parallel_state
from megatron.core.distributed import initialize_distributed
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.optimizer import OptimizerConfig, LRSchedulerConfig, get_optimizer_and_lr_scheduler

# 导入我们训练脚本中的组件
from training_deepseek_v3_mcore import (
    DeepseekV3MegatronModel,
    setup_deepseek_v3_config,
    forward_step
)

def get_test_args():
    """设置测试参数"""
    parser = argparse.ArgumentParser(description='Test DeepSeek V3 Megatron Core Implementation')
    
    # 小型模型配置
    parser.add_argument('--num-layers', type=int, default=2, help='模型层数')
    parser.add_argument('--hidden-size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--num-attention-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num-key-value-heads', type=int, default=4, help='键值注意力头数量')
    parser.add_argument('--intermediate-size', type=int, default=256, help='中间层大小')
    parser.add_argument('--max-position-embeddings', type=int, default=128, help='最大位置编码长度')
    parser.add_argument('--vocab-size', type=int, default=1000, help='词表大小')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--seq-length', type=int, default=16, help='序列长度')
    
    # MoE设置
    parser.add_argument('--use-moe', action='store_true', help='是否使用MoE')
    parser.add_argument('--num-experts', type=int, default=2, help='MoE专家数量')
    parser.add_argument('--experts-per-token', type=int, default=1, help='每个token使用的专家数量')
    parser.add_argument('--moe-layer-freq', type=int, default=2, help='MoE层的频率')
    
    # 分布式设置
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='张量模型并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='流水线模型并行大小')
    
    # 其他参数
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--local_rank', type=int, default=None, help='本地进程排名')
    
    args = parser.parse_args()
    
    # 添加优化器和学习率参数
    args.lr = 1e-4
    args.min_lr = 1e-5
    args.weight_decay = 0.01
    args.clip_grad = 1.0
    args.lr_decay_style = 'linear'
    args.lr_warmup_iters = 5
    
    # 设置padding token
    args.pad_token_id = 0
    
    return args

def create_synthetic_data(args):
    """创建合成输入数据"""
    # 随机输入ID
    input_ids = torch.randint(
        1, args.vocab_size, (args.batch_size, args.seq_length), 
        dtype=torch.long, device='cuda'
    )
    
    # 创建注意力掩码 (全1表示没有padding)
    attention_mask = torch.ones(
        (args.batch_size, args.seq_length),
        dtype=torch.long, device='cuda'
    )
    
    # 随机添加一些padding作为测试
    if args.batch_size > 1:
        # 为第一个样本添加一些padding
        pad_length = args.seq_length // 4
        input_ids[0, -pad_length:] = args.pad_token_id
        attention_mask[0, -pad_length:] = 0
    
    # 创建位置ID
    position_ids = torch.arange(
        args.seq_length, dtype=torch.long, device='cuda'
    ).unsqueeze(0).expand(args.batch_size, -1)
    
    # 将pad位置的position_id设为0
    position_ids = position_ids * attention_mask
    
    # 使用input_ids作为标签 (自回归任务)
    labels = input_ids.clone()
    
    return input_ids, attention_mask, position_ids, labels

def test_forward_backward(model, args):
    """测试模型的前向和后向传播"""
    print("创建合成数据...")
    input_ids, attention_mask, position_ids, labels = create_synthetic_data(args)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")
    print(f"位置ID形状: {position_ids.shape}")
    
    print("\n执行前向传播...")
    model.train()
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels
    )
    
    loss = outputs[0]
    logits = outputs[1]
    
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
    
    print("\n执行反向传播...")
    loss.backward()
    
    # 检查梯度是否存在并且不是NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm = param.grad.data.norm(2) if param.grad is not None else 0
            print(f"{name}: 参数范数={param.data.norm(2):.4f}, 梯度范数={grad_norm:.4f}")
    
    return loss.item()

def test_optimizer(model, args):
    """测试优化器"""
    print("\n设置优化器...")
    
    # 优化器配置
    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        use_cpu_initialization=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad
    )
    
    # 学习率调度器配置
    lr_scheduler_config = LRSchedulerConfig(
        lr_decay_style=args.lr_decay_style,
        warmup_steps=args.lr_warmup_iters,
        min_lr=args.min_lr,
        total_steps=100,  # 假设总共训练100步
    )
    
    # 创建优化器和学习率调度器
    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
        model=model,
        config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # 测试优化器步骤
    print("\n执行优化器步骤...")
    input_ids, attention_mask, position_ids, labels = create_synthetic_data(args)
    
    # 模拟一个小的训练循环
    losses = []
    for i in range(5):  # 运行5步
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels
        )
        
        loss = outputs[0]
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 优化器步骤
        optimizer.step()
        lr_scheduler.step()
        
        print(f"步骤 {i+1}: Loss = {loss.item():.6f}, LR = {lr_scheduler.get_lr():.8f}")
    
    return losses

def main():
    """主测试函数"""
    args = get_test_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
        
    # 初始化分布式环境
    try:
        initialize_distributed(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            world_size=None,  # 自动检测
            rank=None,  # 自动检测
            distributed_port=None,
            distributed_backend='nccl' if torch.cuda.is_available() else 'gloo'
        )
        print(f"分布式初始化成功: 世界大小={parallel_state.get_data_parallel_world_size()}")
    except Exception as e:
        print(f"分布式初始化警告: {e}")
        print("继续单GPU测试...")
    
    # 设置CUDA随机种子
    model_parallel_cuda_manual_seed(args.seed)
    
    print("\n设置DeepSeek V3配置...")
    config = setup_deepseek_v3_config(args)
    
    print("\n创建模型...")
    model = DeepseekV3MegatronModel(
        config=config,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_position_embeddings
    )
    
    # 打印模型结构摘要
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型大小: {total_params:,} 参数")
    print(f"可训练参数: {trainable_params:,} 参数")
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        model = model.to(device)
    
    # 测试前向和后向传播
    print("\n===== 测试前向和后向传播 =====")
    loss = test_forward_backward(model, args)
    
    # 测试优化器
    print("\n===== 测试优化器 =====")
    losses = test_optimizer(model, args)
    
    # 测试结果摘要
    print("\n===== 测试结果摘要 =====")
    print(f"初始Loss: {loss:.6f}")
    print(f"优化后Loss: {losses[-1]:.6f}")
    print(f"Loss变化: {(losses[-1] - loss):.6f} ({(losses[-1] - loss) / loss * 100:.2f}%)")
    
    if losses[-1] < loss:
        print("\n✓ 测试成功: Loss正在减小，实现似乎正常工作!")
    else:
        print("\n⚠ 警告: Loss没有减小，可能需要检查实现!")
    
    # 清理
    try:
        parallel_state.destroy_model_parallel()
        print("\n清理分布式环境完成")
    except Exception as e:
        print(f"\n清理分布式环境时出错: {e}")

if __name__ == "__main__":
    main() 