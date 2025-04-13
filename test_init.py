#!/usr/bin/env python
# coding=utf-8

import os
import torch
from megatron.core import parallel_state
import argparse

def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, 
                          world_size=None, rank=None, distributed_port=None, 
                          distributed_backend='nccl', master_addr='localhost', master_port='6000'):
    """初始化分布式训练环境
    
    Args:
        tensor_model_parallel_size: 张量模型并行大小
        pipeline_model_parallel_size: 流水线模型并行大小
        world_size: 总进程数，如果为None则自动获取
        rank: 当前进程序号，如果为None则自动获取
        distributed_port: 分布式训练端口
        distributed_backend: 分布式后端，默认为'nccl'
        master_addr: 主节点IP地址，默认为'localhost'
        master_port: 主节点端口，默认为'6000'
    """
    # 清理旧的模型并行状态
    parallel_state.destroy_model_parallel()
    
    # 设置分布式环境变量
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = master_addr
        print(f"已设置 MASTER_ADDR={master_addr}")
    
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = master_port
        print(f"已设置 MASTER_PORT={master_port}")
    
    # 如果未指定rank和world_size，使用环境变量获取
    if rank is None:
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        elif 'LOCAL_RANK' in os.environ:
            rank = int(os.environ['LOCAL_RANK'])
        else:
            rank = 0
            os.environ['RANK'] = str(rank)
            print(f"已设置 RANK={rank}")
            
    if world_size is None:
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = torch.cuda.device_count()
            os.environ['WORLD_SIZE'] = str(world_size)
            print(f"已设置 WORLD_SIZE={world_size}")
    
    # 设置当前设备
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # 初始化进程组
    init_method = 'env://'
    if distributed_port is not None:
        init_method = f'tcp://{master_addr}:{distributed_port}'
    
    # 打印环境变量
    print("\n分布式环境变量:")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', '未设置')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', '未设置')}")
    print(f"RANK: {os.environ.get('RANK', '未设置')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', '未设置')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', '未设置')}")
    print(f"初始化方法: {init_method}")
    
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
    parser.add_argument('--master-addr', default='localhost', help='主节点IP地址')
    parser.add_argument('--master-port', default='6000', help='主节点端口')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = get_args()
    
    print("===== 初始化前状态 =====")
    print(f"检查CUDA是否可用: {torch.cuda.is_available()}")
    
    # 添加CUDA可用性检查
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU模式运行")
        print("测试跳过，请确保CUDA环境正确配置!")
        return
        
    try:
        gpu_count = torch.cuda.device_count()
        print(f"可见GPU数量: {gpu_count}")
        
        if gpu_count == 0:
            print("警告: 没有检测到可用的GPU，将使用CPU模式运行")
            print("测试跳过，请确保CUDA环境正确配置!")
            return
            
        # 获取当前设备信息
        current_device = torch.cuda.current_device()
        print(f"当前活跃设备: {current_device}")
        
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前设备名称: {device_name}")
        
        # 尝试一个简单的GPU操作
        test_tensor = torch.tensor([1.0, 2.0], device=f"cuda:{current_device}")
        print(f"测试GPU访问成功: {test_tensor.device}")
    except Exception as e:
        print(f"CUDA初始化失败: {str(e)}")
        print("测试跳过，请检查CUDA环境!")
        return
    
    # 初始化分布式环境 - 使用try/except处理
    try:
        print("\n开始初始化分布式环境...")
        initialize_distributed(
            tensor_model_parallel_size=args.tensor_model_parallel_size, 
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            distributed_backend=args.distributed_backend,
            master_addr=args.master_addr,
            master_port=args.master_port
        )
        
        print("\n===== 初始化后状态 =====")
        # 打印更多详细信息
        print(f"分布式初始化成功!")
        print(f"Process rank: {torch.distributed.get_rank()}, "
            f"World size: {torch.distributed.get_world_size()}, "
            f"Device: {torch.cuda.current_device()}")
        
        print("\n===== 模型并行状态 =====")
        from megatron.core import parallel_state
        print(f"数据并行大小: {parallel_state.get_data_parallel_world_size()}")
        print(f"数据并行等级: {parallel_state.get_data_parallel_rank()}")
        print(f"张量模型并行大小: {parallel_state.get_tensor_model_parallel_world_size()}")
        print(f"张量模型并行等级: {parallel_state.get_tensor_model_parallel_rank()}")
        print(f"流水线模型并行大小: {parallel_state.get_pipeline_model_parallel_world_size()}")
        print(f"流水线模型并行等级: {parallel_state.get_pipeline_model_parallel_rank()}")
        
        # 创建一个简单的张量并验证设备
        test_tensor = torch.randn(2, 2).cuda()
        print(f"\n测试张量设备: {test_tensor.device}")
        
        # 清理分布式环境
        print("\n===== 清理环境 =====")
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("分布式进程组已销毁")
    except Exception as e:
        print(f"分布式初始化失败: {str(e)}")
    
    print("测试完成!")

if __name__ == "__main__":
    main() 