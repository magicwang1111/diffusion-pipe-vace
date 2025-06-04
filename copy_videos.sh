#!/bin/bash

# 源目录
SRC_DIR="/mnt/data/wangxi/diffusion-pipe/dataset/20250416医美处理后/normalized_videos"
# 目标目录
DEST_DIR="/mnt/data/wangxi/diffusion-pipe/dataset/20250416医美处理后/视频"

# 支持的视频文件扩展名
EXTS=("mp4" "mov" "avi" "mkv" "flv")

# 初始化计数器
count=0

# 收集所有视频路径
file_list=()
for ext in "${EXTS[@]}"; do
  while IFS= read -r -d '' filepath; do
    file_list+=("$filepath")
  done < <(find "$SRC_DIR" -type f -name "*.${ext}" -print0)
done

# 遍历并复制
for filepath in "${file_list[@]}"; do
  filename=$(basename "$filepath")
  dest="$DEST_DIR/$filename"

  # 如果文件已存在，加后缀避免覆盖
  if [ -e "$dest" ]; then
    base="${filename%.*}"
    extn="${filename##*.}"
    i=1
    while [ -e "$DEST_DIR/${base}_$i.$extn" ]; do
      ((i++))
    done
    dest="$DEST_DIR/${base}_$i.$extn"
  fi

  echo "Copying $filepath -> $dest"
  cp "$filepath" "$dest"
  ((count++))
done

echo "✅ 总共复制了 $count 个视频文件。"
