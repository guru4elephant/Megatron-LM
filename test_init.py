#!/usr/bin/env python
# coding=utf-8

import os
import torch
from megatron.core import parallel_state
import argparse

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

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test distributed initialization')
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='张量模型并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='流水线模型并行大小')
    parser.add_argument('--distributed-backend', default='nccl', choices=['nccl', 'gloo'], help='分布式后端')
    parser.add_argument('--local_rank', type=int, default=None, help='本地进程排名(由torch.distributed自动提供)')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = get_args()
    
    # 初始化分布式环境
    initialize_distributed(
        tensor_model_parallel_size=args.tensor_model_parallel_size, 
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        distributed_backend=args.distributed_backend
    )
    
    # 打印当前进程信息
    print(f"Process rank: {torch.distributed.get_rank()}, "
          f"World size: {torch.distributed.get_world_size()}, "
          f"Device: {torch.cuda.current_device()}")
    
    # 清理分布式环境
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 