# Megatron-LM 训练示例使用指南

本文档提供了如何在单机多卡和多机多卡环境下运行 `megatron_core_training_example.py` 训练脚本的详细说明。

## 环境准备

确保已安装以下依赖：

```bash
pip install torch numpy wandb
```

## 单机多卡训练

### 基本用法

使用 `torch.distributed.launch` 或 `torchrun` 启动分布式训练：

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=TP_SIZE \
    --pipeline-model-parallel-size=PP_SIZE \
    [其他参数]

# 或使用 torchrun (推荐)
torchrun --nproc_per_node=NUM_GPUS \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=TP_SIZE \
    --pipeline-model-parallel-size=PP_SIZE \
    [其他参数]
```

其中：
- `NUM_GPUS`: 使用的 GPU 数量
- `TP_SIZE`: 张量并行度
- `PP_SIZE`: 流水线并行度

注意：确保 `NUM_GPUS = TP_SIZE * PP_SIZE * DP_SIZE`，其中 `DP_SIZE` 是数据并行度。

### 单机 Tensor 并行示例

在一台有 8 个 GPU 的机器上运行 2-way 张量并行：

```bash
torchrun --nproc_per_node=8 \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=1 \
    --batch-size=16 \
    --epochs=3 \
    --train-steps=1000 \
    --sequence-parallel \
    --fp16
```

### 单机 Pipeline 并行示例

在一台有 8 个 GPU 的机器上运行 2-way 流水线并行：

```bash
torchrun --nproc_per_node=8 \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=1 \
    --pipeline-model-parallel-size=2 \
    --batch-size=16 \
    --epochs=3 \
    --train-steps=1000 \
    --fp16
```

### 单机混合并行示例

在一台有 8 个 GPU 的机器上运行 2-way 张量并行 + 2-way 流水线并行（共 4 个模型副本，每个副本使用 2x2=4 个 GPU）：

```bash
torchrun --nproc_per_node=8 \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=2 \
    --batch-size=16 \
    --epochs=3 \
    --train-steps=1000 \
    --sequence-parallel \
    --fp16
```

### 单机调试技巧

1. 使用较小的模型和数据集进行快速测试：

```bash
torchrun --nproc_per_node=2 \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=1 \
    --num-layers=2 \
    --hidden-size=128 \
    --num-attention-heads=4 \
    --batch-size=4 \
    --train-steps=10 \
    --dataset-size=100 \
    --monitor-memory-continuously
```

2. 开启内存监控：

```bash
# 添加以下参数来监控内存使用
--monitor-memory-continuously --memory-monitor-interval=0.5
```

3. 使用较短的序列长度和较小的批次大小进行初始测试：

```bash
--seq-length=128 --batch-size=4
```

## 多机多卡训练

### 基本设置

在多机环境下，需要指定：
- 世界大小（所有机器上的总 GPU 数量）
- 每台机器的 rank
- 初始化方法 URL（master 节点的地址和端口）

### 示例启动命令

假设有两台机器，每台 4 个 GPU，总共 8 个 GPU：

**第一台机器 (Master, Rank 0):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=MASTER_PORT \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=2 \
    --batch-size=16 \
    --epochs=3 \
    --train-steps=1000 \
    --init-method=tcp://MASTER_IP:MASTER_PORT \
    --world-size=8 \
    --fp16
```

**第二台机器 (Rank 1):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=MASTER_PORT \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=2 \
    --batch-size=16 \
    --epochs=3 \
    --train-steps=1000 \
    --init-method=tcp://MASTER_IP:MASTER_PORT \
    --world-size=8 \
    --fp16
```

将 `MASTER_IP` 替换为主节点的 IP 地址，`MASTER_PORT` 替换为可用端口（例如 29500）。

### 使用 Slurm 启动多机训练

如果在 Slurm 集群上运行，可以使用以下脚本：

```bash
#!/bin/bash
#SBATCH --job-name=megatron_train
#SBATCH --nodes=2               # 请求 2 个节点
#SBATCH --ntasks-per-node=1     # 每个节点 1 个任务
#SBATCH --cpus-per-task=12      # 每个任务 12 个 CPU 核心
#SBATCH --gres=gpu:4            # 每个节点 4 个 GPU
#SBATCH --time=12:00:00         # 最大运行时间
#SBATCH --output=%x-%j.out      # 输出文件格式

# 获取主节点信息
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# 设置每个节点上的进程数
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

# 计算总 GPU 数
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# 在每个节点上启动训练
srun --jobid $SLURM_JOBID bash -c "
    torchrun --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        megatron_core_training_example.py \
        --tensor-model-parallel-size=2 \
        --pipeline-model-parallel-size=2 \
        --batch-size=16 \
        --epochs=3 \
        --train-steps=1000 \
        --init-method=tcp://$MASTER_ADDR:$MASTER_PORT \
        --world-size=$WORLD_SIZE \
        --fp16
"
```

### 多机调试技巧

1. 从单节点单 GPU 开始，逐步增加规模：

```bash
# 在单 GPU 上进行测试
CUDA_VISIBLE_DEVICES=0 python megatron_core_training_example.py \
    --tensor-model-parallel-size=1 --pipeline-model-parallel-size=1 
```

2. 使用更小的模型和批次进行多机连通性测试：

```bash
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=MASTER_PORT \
    megatron_core_training_example.py \
    --tensor-model-parallel-size=2 --pipeline-model-parallel-size=1 \
    --num-layers=2 --hidden-size=128 --train-steps=10 --batch-size=4
```

3. 检查分布式初始化是否成功：
   - 所有节点应该能够连接到主节点
   - 检查所有节点上的防火墙是否允许指定端口的通信
   - 确保 `--world-size` 参数与实际使用的 GPU 总数匹配

4. 使用 `--save-performance-summary` 选项保存性能数据进行分析

## 重要参数说明

### 模型参数
- `--num-layers`: Transformer 的层数
- `--hidden-size`: 隐藏层大小
- `--num-attention-heads`: 注意力头数
- `--ffn-hidden-size`: 前馈网络隐藏层大小
- `--vocab-size`: 词汇表大小
- `--seq-length`: 序列长度

### 训练参数
- `--batch-size`: 批处理大小
- `--epochs`: 训练轮数
- `--train-steps`: 训练步数
- `--lr`: 学习率
- `--warmup-steps`: 预热步数
- `--gradient-clip-val`: 梯度裁剪值

### 并行化参数
- `--tensor-model-parallel-size`: 模型张量并行度
- `--pipeline-model-parallel-size`: 模型流水线并行度
- `--sequence-parallel`: 启用序列并行

### 分布式参数
- `--world-size`: 总进程数（总 GPU 数）
- `--init-method`: 分布式初始化方法（如 `tcp://localhost:12345`）

### 精度参数
- `--fp16`: 使用 FP16 精度
- `--bf16`: 使用 BF16 精度

### 监控参数
- `--monitor-memory-continuously`: 持续监控内存使用情况
- `--save-performance-summary`: 保存性能摘要

## 常见问题解决

1. **分布式初始化失败**
   - 检查所有节点网络连接
   - 确保指定的端口未被防火墙阻止
   - 检查 `--init-method` URL 格式是否正确

2. **显存不足 (OOM)**
   - 减小批次大小 `--batch-size`
   - 减小模型大小（`--hidden-size`, `--num-layers`）
   - 减小序列长度 `--seq-length`
   - 使用 `--fp16` 或 `--bf16` 降低精度

3. **并行化配置错误**
   - 确保 `NUM_GPUS = TP_SIZE * PP_SIZE * DP_SIZE`
   - 张量并行度必须能整除注意力头数 (`num-attention-heads` % `tensor-model-parallel-size` == 0)

4. **进程间通信超时**
   - 增加 NCCL 超时设置: `export NCCL_TIMEOUT=3600`
   - 设置较松的 socket 超时: `export NCCL_SOCKET_NTHREADS=1`
   - 尝试不同的网络接口: `export NCCL_SOCKET_IFNAME=eth0`

5. **性能不佳**
   - 使用 `--monitor-memory-continuously` 检查内存使用情况
   - 尝试不同的批次大小和并行化配置
   - 检查数据加载瓶颈（增加 `--num-workers`） 