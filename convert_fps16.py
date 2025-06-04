import os
import subprocess
import tempfile
import shutil

input_dir = "/home/wangxi/diffusion-pipe/dataset/20250526医美训练素材整理fps16"
video_exts = (".mp4", ".mov", ".avi", ".mkv")

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(video_exts):
            input_path = os.path.join(root, file)

            # 创建临时输出路径（同目录）
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=root)
            os.close(tmp_fd)  # 关闭文件描述符

            cmd = [
                "ffmpeg",
                "-y",                # 覆盖模式
                "-i", input_path,
                "-r", "16",          # 目标帧率
                "-c:v", "libx264",   # 编码器
                "-preset", "fast",
                "-crf", "17",        # 质量
                "-pix_fmt", "yuv420p",
                tmp_path
            ]

            print(f"▶ 正在转码: {input_path}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"❌ 转码失败: {input_path}\n{result.stderr.decode()}")
                os.remove(tmp_path)
            else:
                shutil.move(tmp_path, input_path)
                print(f"✅ 替换成功: {input_path}")
