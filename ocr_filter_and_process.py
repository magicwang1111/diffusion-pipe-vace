#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import shutil
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import easyocr

# === 配置参数 ===
INPUT_DIR = Path("/data/wangxi/aimanga-video-dataset/dataset/20250421医美/outputs")
OUTPUT_DIR = Path("/data/wangxi/aimanga-video-dataset/dataset/20250421医美/easyocr_filter/")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=True)
TARGET_FRAMES = 97  # Fixed target frame count
SAMPLE_COUNT = 2
FPS_OUT = 24
RESOLUTION = 921600
OCR_CONFIDENCE_THRESHOLD = 0.3
MIN_TEXT_REGIONS = 1
MIN_TEXT_FRAMES = 1

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def has_text(video: Path) -> bool:
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return False
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // SAMPLE_COUNT)
        frames_with_text = 0
        
        for frame_idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, fr = cap.read()
            if not ret:
                break
                
            if fr is not None and fr.size > 0:
                try:
                    text_results = OCR_READER.readtext(fr, detail=1)
                    confident_texts = [t for t in text_results if t[2] >= OCR_CONFIDENCE_THRESHOLD]
                    if len(confident_texts) >= MIN_TEXT_REGIONS:
                        frames_with_text += 1
                        if frames_with_text >= MIN_TEXT_FRAMES:
                            cap.release()
                            return True
                except:
                    continue
                    
        cap.release()
        return False
    except:
        return False

def evenly_spaced_indices(total: int, n: int) -> list:
    if total <= n:
        # If total frames are less than target, we'll need to duplicate some frames
        return list(range(total))
    
    # For videos with more frames than target, select evenly spaced frames
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]

def resize_frame(frame: np.ndarray, target_pixels: int = RESOLUTION) -> np.ndarray:
    h, w = frame.shape[:2]
    ratio = w / h
    new_h = int(round(((target_pixels / ratio) ** 0.5) / 16) * 16)
    new_w = int(round((new_h * ratio) / 16) * 16)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def process_video(video: Path, out: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
        
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate which frames to include in output
    if total < TARGET_FRAMES:
        # For shorter videos, we'll need to duplicate frames
        # Example: If we have 50 frames and need 97, each source frame
        # will be used approximately 97/50 times (some more than others)
        idxs = []
        for i in range(TARGET_FRAMES):
            # Map each output frame index to a source frame index
            src_idx = min(int(i * total / TARGET_FRAMES), total - 1)
            idxs.append(src_idx)
    else:
        # For longer videos, pick evenly spaced frames
        idxs = evenly_spaced_indices(total, TARGET_FRAMES)
    
    ensure_dir(out.parent)
    
    # Read all frames we'll need
    frames = {}
    for idx in set(idxs):  # Use set to avoid reading the same frame multiple times
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = resize_frame(frame)
    
    # Create output video
    if frames:
        h, w = next(iter(frames.values())).shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out), fourcc, FPS_OUT, (w, h))
        
        # Write frames in the correct order
        for idx in idxs:
            if idx in frames:
                vw.write(frames[idx])
        
        vw.release()
    
    cap.release()

def main():
    video_files = [p for p in INPUT_DIR.rglob('*') if p.suffix.lower() in VIDEO_EXTS]
    print(f"Found {len(video_files)} videos for OCR filtering.")
    
    for video in tqdm(video_files, desc="Processing"):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if has_text(video):
            print(f"Skipping {video.name} (detected text)")
            continue
            
        out_path = OUTPUT_DIR / video.relative_to(INPUT_DIR)
        process_video(video, out_path)
        
        # Verify the output has exactly 97 frames
        if out_path.exists():
            verify_cap = cv2.VideoCapture(str(out_path))
            out_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            verify_cap.release()
            
            if out_frames != TARGET_FRAMES:
                print(f"Warning: Output video {out_path.name} has {out_frames} frames instead of {TARGET_FRAMES}")

if __name__ == "__main__":
    main()