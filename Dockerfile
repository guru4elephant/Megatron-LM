FROM nvcr.io/nvidia/pytorch:23.12-py3

# 设置工作目录
WORKDIR /workspace

# 接收代理环境变量
ARG http_proxy
ARG https_proxy

# 设置代理环境变量
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

# 安装基础软件包
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    tmux \
    htop \
    emacs \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 克隆Megatron-LM特定版本
RUN git clone -b v0.11.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM

# 安装Megatron-LM依赖
WORKDIR /workspace/Megatron-LM

# 安装基础依赖
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements-dev.txt

# 安装wandb（用于实验跟踪）
RUN pip install --no-cache-dir wandb

# 安装Megatron-Core
RUN pip install --no-cache-dir -e .

# 清除代理环境变量
ENV http_proxy=
ENV https_proxy=

# 设置环境变量
ENV PYTHONPATH=/workspace/Megatron-LM:$PYTHONPATH
ENV NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置工作目录
WORKDIR /workspace

# 将示例训练脚本复制到容器中
COPY megatron_core_training_example.py /workspace/
COPY training_example_readme.md /workspace/

# 添加NVIDIA visible devices环境变量
ENV NVIDIA_VISIBLE_DEVICES=all

# 设置容器启动命令
CMD ["/bin/bash"] 