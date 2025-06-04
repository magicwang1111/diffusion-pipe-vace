#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import easyocr

# 与原逻辑保持一致的参数
OCR_CONFIDENCE_THRESHOLD = 0.3
MIN_TEXT_REGIONS = 2
MIN_TEXT_FRAMES = 1
SAMPLE_COUNT = 2  # 总共抽取多少帧

def has_subtitles(video_path: str) -> bool:
    """
    按原有逻辑检测视频是否包含字幕：
    - 均匀抽取 SAMPLE_COUNT 帧，
    - 对每帧执行 reader.readtext(frame, detail=1)，
    - 统计置信度 ≥ OCR_CONFIDENCE_THRESHOLD 的文字区域，
    - 若累计达到 MIN_TEXT_FRAMES，则视为有字幕。
    """
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}", file=sys.stderr)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    # 计算均匀采样间隔
    step = SAMPLE_COUNT
    frames_with_text = 0

    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # detail=1 保留置信度信息
        try:
            results = reader.readtext(frame, detail=1)  # EasyOCR 默认模式&#8203;:contentReference[oaicite:2]{index=2}
        except Exception as e:
            print(f"OCR 错误（帧 {idx}）: {e}", file=sys.stderr)
            continue

        # 筛选高置信度区域
        confident = [r for r in results if r[2] >= OCR_CONFIDENCE_THRESHOLD]
        if len(confident) >= MIN_TEXT_REGIONS:
            frames_with_text += 1
            if frames_with_text >= MIN_TEXT_FRAMES:
                cap.release()
                return True

    cap.release()
    return False

if __name__ == "__main__":
    # 直接在脚本中测试该文件
    test_file = "/mnt/data/wangxi/diffusion-pipe/dataset/20250417chi处理后v4/normalized_videos/ChiikawaSub_scenes/ChiikawaSub_scene_0068.mp4"
    if has_subtitles(test_file):
        print("检测到字幕（文字），此视频应被过滤。")
    else:
        print("未检测到字幕（文字），此视频未被过滤。")
