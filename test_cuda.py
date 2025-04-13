#!/usr/bin/env python
# coding=utf-8

import torch
import os

def main():
    """测试CUDA是否正常工作"""
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
    
    print(f"\n===== 环境变量 =====")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
    print(f"\n===== GPU测试 =====")
    try:
        device = torch.device("cuda:0")
        x = torch.ones(5, 3, device=device)
        y = torch.ones(5, 3, device=device) * 2
        z = x + y
        print(f"GPU计算测试成功: {z.mean().item()}")
    except Exception as e:
        print(f"GPU测试失败: {str(e)}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 