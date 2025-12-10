#!/bin/bash

# AI服务启动脚本

echo "=========================================="
echo "Bridge Inspection AI Service"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: Not in a virtual environment"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查依赖是否安装
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "Error: Dependencies not installed. Please run:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "weights/best.pt" ]; then
    echo "Error: Model weights not found at weights/best.pt"
    exit 1
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "Warning: config.yaml not found, using defaults"
fi

# 创建日志目录
mkdir -p logs

# 获取端口号（从配置文件或使用默认值）
PORT=${PORT:-5000}

echo "Starting AI service on port $PORT..."
echo "=========================================="

# 启动服务
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info

# 或者直接运行
# python3 app/main.py
