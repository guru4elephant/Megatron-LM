#!/usr/bin/env python
# coding=utf-8

import os
import torch
import argparse

def initialize_distributed(backend='nccl', master_addr='localhost', master_port='6000', 
                          rank=None, world_size=None, local_rank=None, distributed_port=None):
    """初始化分布式训练环境
    
    Args:
        backend: 分布式后端，默认为'nccl'
        master_addr: 主节点IP地址，默认为'localhost'
        master_port: 主节点端口，默认为'6000'
        rank: 当前进程序号，如果为None则自动获取
        world_size: 总进程数，如果为None则自动获取
        local_rank: 本地进程序号，如果为None则自动获取
        distributed_port: 分布式训练端口，如果提供则使用tcp初始化方法
    """
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
    if local_rank is None:
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = rank % torch.cuda.device_count()
            os.environ['LOCAL_RANK'] = str(local_rank)
            print(f"已设置 LOCAL_RANK={local_rank}")
    
    device = local_rank
    torch.cuda.set_device(device)
    
    # 初始化进程组
    init_method = 'env://'
    if distributed_port is not None:
        init_method = f'tcp://{master_addr}:{distributed_port}'
    
    print("\n分布式环境变量:")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', '未设置')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', '未设置')}")
    print(f"RANK: {os.environ.get('RANK', '未设置')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', '未设置')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', '未设置')}")
    print(f"初始化方法: {init_method}")
    print(f"后端: {backend}")
    
    # 初始化pytorch分布式
    if not torch.distributed.is_initialized():
        print("\n开始初始化分布式环境...")
        try:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method
            )
            print("分布式初始化成功!")
            print(f"进程等级: {torch.distributed.get_rank()}, "
                  f"总进程数: {torch.distributed.get_world_size()}, "
                  f"当前设备: {torch.cuda.current_device()}")
            return True
        except Exception as e:
            print(f"分布式初始化失败: {str(e)}")
            return False
    else:
        print("分布式环境已经初始化")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分布式环境初始化工具')
    parser.add_argument('--backend', type=str, default='nccl', help='分布式后端 (nccl, gloo, etc.)')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点IP地址')
    parser.add_argument('--master_port', type=str, default='6000', help='主节点端口')
    parser.add_argument('--rank', type=int, default=None, help='当前进程序号')
    parser.add_argument('--world_size', type=int, default=None, help='总进程数')
    parser.add_argument('--local_rank', type=int, default=None, help='本地进程序号')
    parser.add_argument('--distributed_port', type=str, default=None, help='分布式训练端口')
    
    args = parser.parse_args()
    
    # 输出CUDA信息
    print("===== CUDA检查 =====")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用!")
        return
    
    print(f"\n===== GPU信息 =====")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 初始化分布式环境
    success = initialize_distributed(
        backend=args.backend,
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        world_size=args.world_size,
        local_rank=args.local_rank,
        distributed_port=args.distributed_port
    )
    
    if success:
        # 简单的分布式通信测试
        try:
            if torch.distributed.get_world_size() > 1:
                print("\n===== 分布式通信测试 =====")
                tensor = torch.ones(1).cuda() * torch.distributed.get_rank()
                print(f"发送张量: {tensor.item()}")
                
                torch.distributed.all_reduce(tensor)
                expected_result = sum(range(torch.distributed.get_world_size()))
                print(f"接收张量: {tensor.item()}, 预期结果: {expected_result}")
                
                if tensor.item() == expected_result:
                    print("分布式通信测试成功!")
                else:
                    print("分布式通信测试失败!")
        except Exception as e:
            print(f"分布式通信测试失败: {str(e)}")
    
    # 清理分布式环境
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        print("\n分布式环境已清理")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 