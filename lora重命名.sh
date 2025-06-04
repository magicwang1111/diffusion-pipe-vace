#!/bin/bash

# 设置源目录和目标目录
SOURCE_DIR="/mnt/data/wangxi/diffusion-pipe-vace/output/20250528医美人工标注i2v-fps24/20250529_03-20-35"
TARGET_DIR="/mnt/data/wangxi/diffusion-pipe-vace/output/20250528医美人工标注i2vfps24整理"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 查找并处理所有 adapter_model.safetensors
find "$SOURCE_DIR" -name "adapter_model.safetensors" | while read -r filepath; do
    # 提取父目录名作为标识（如 epoch10）
    parent_dir=$(basename "$(dirname "$filepath")")
    # 构建新的文件名
    new_filename="20250528医美人工标注i2vfps24-${parent_dir}.safetensors"
    # 拷贝并改名到目标目录
    cp "$filepath" "$TARGET_DIR/$new_filename"
    echo "Copied: $filepath → $TARGET_DIR/$new_filename"
done
