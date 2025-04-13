#!/bin/bash

# 设置代理环境变量
export http_proxy=
export https_proxy=
echo "已设置代理: http_proxy=$http_proxy, https_proxy=$https_proxy"

# 设置镜像名称和标签
IMAGE_NAME="megatron-lm"
TAG="v0.11.0"

# 检查megatron_core_training_example.py文件是否存在
if [ ! -f "megatron_core_training_example.py" ]; then
    echo "错误: megatron_core_training_example.py 文件不存在!"
    echo "请确保该文件与Dockerfile在同一目录下。"
    exit 1
fi

# 检查training_example_readme.md文件是否存在
if [ ! -f "training_example_readme.md" ]; then
    echo "错误: training_example_readme.md 文件不存在!"
    echo "请确保该文件与Dockerfile在同一目录下。"
    exit 1
fi

# 构建Docker镜像，传递代理环境变量
echo "开始构建 $IMAGE_NAME:$TAG 镜像..."
docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    -t $IMAGE_NAME:$TAG .

# 检查构建是否成功
if [ $? -eq 0 ]; then
    echo "镜像构建成功: $IMAGE_NAME:$TAG"
    echo "可用以下命令运行容器:"
    echo "docker run --gpus all -it --rm $IMAGE_NAME:$TAG"
    echo "如需挂载当前目录，可使用:"
    echo "docker run --gpus all -it --rm -v $(pwd):/workspace/mount $IMAGE_NAME:$TAG"
else
    echo "镜像构建失败!"
    exit 1
fi

# 打标签latest版本
docker tag $IMAGE_NAME:$TAG $IMAGE_NAME:latest
echo "镜像也被标记为 $IMAGE_NAME:latest"

# 清除代理设置
unset http_proxy
unset https_proxy
echo "已清除代理设置" 