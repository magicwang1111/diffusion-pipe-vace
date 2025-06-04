#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys 
import shutil
import subprocess 
import tempfile
from pathlib import Path
from typing import List, Tuple, Set, Optional
import logging

import cv2
import numpy as np
from tqdm import tqdm
import easyocr
from scenedetect import open_video, SceneManager, ContentDetector

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_processor")

OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=True)
INPUT_DIR = Path("/home/wangxi/diffusion-pipe/dataset/20250417chi")
OUTPUT_DIR = Path("/home/wangxi/diffusion-pipe/dataset/20250417chi处理后v3")
BUCKETS = [121]
TARGET_BUCKET = min(BUCKETS)
FPS_OUT = 24
RESOLUTION = 921600
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# Adjust scene detection sensitivity
CONTENT_THRESHOLD = 10
MIN_SCENE_LEN = int(TARGET_BUCKET)
TRANSITION_BUFFER = 5

# Maximum recursion depth
MAX_RECURSION_DEPTH = 3
MIN_SEGMENT_LENGTH = 0.8 * TARGET_BUCKET
MAX_SEGMENT_LENGTH = 1.2 * TARGET_BUCKET

# OCR settings
OCR_CONFIDENCE_THRESHOLD = 0.4
MIN_TEXT_REGIONS = 1
MIN_TEXT_FRAMES = 1

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
if "--ffmpeg" in sys.argv:
    idx = sys.argv.index("--ffmpeg")
    if idx + 1 < len(sys.argv):
        FFMPEG_BIN = sys.argv[idx + 1]
OCR_ENABLED = "--no-ocr" not in sys.argv

# Check available encoders
def get_available_encoders() -> Set[str]:
    try:
        cmd = [FFMPEG_BIN, "-encoders"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        encoders = set()
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith(" V"):  # Video encoders
                parts = line.split()
                if len(parts) > 1:
                    encoders.add(parts[1])
        return encoders
    except Exception as e:
        logger.error(f"[get_available_encoders] error: {e}")
        return set()

AVAILABLE_ENCODERS = get_available_encoders()
logger.info(f"[init] Available encoders: {AVAILABLE_ENCODERS}")

# Choose codec based on availability
def get_available_codec() -> str:
    if 'libx264' in AVAILABLE_ENCODERS:
        return 'libx264'
    elif 'mpeg4' in AVAILABLE_ENCODERS:
        return 'mpeg4'
    else:
        logger.warning("[get_available_codec] No preferred codecs (libx264, mpeg4) found, falling back to default")
        return 'mpeg4'  # Default fallback, may still fail if not supported

SELECTED_CODEC = get_available_codec()
logger.info(f"[init] Selected codec: {SELECTED_CODEC}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def refine_boundary(cap: cv2.VideoCapture, frame_no: int, buffer: int) -> int:
    best_frame, best_diff = frame_no, float('inf')
    for off in range(-buffer, buffer + 1):
        idx = frame_no + off
        if idx < 1:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret1, f1 = cap.read()
        ret2, f2 = cap.read()
        if not (ret1 and ret2):
            continue
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        diff = np.sum(cv2.absdiff(g1, g2))
        if diff < best_diff:
            best_diff, best_frame = diff, idx
    logger.info(f"      [refine] frame {frame_no} -> {best_frame}")
    return best_frame

def analyze_transitions(path: str, starts: List[int], buffer: int = TRANSITION_BUFFER) -> List[int]:
    logger.info(f"    [analyze_transitions] refining {len(starts)} cuts")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.warning("      [analyze_transitions] cannot open video")
        return starts
    refined = [refine_boundary(cap, s, buffer) for s in starts]
    cap.release()
    return refined

def detect_scenes_simple(path: str) -> List[Tuple[int, int]]:
    logger.info(f"    [detect_scenes_simple] {os.path.basename(path)}")
    try:
        video = open_video(path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(
            threshold=CONTENT_THRESHOLD,
            min_scene_len=MIN_SCENE_LEN,
            luma_only=True
        ))
        manager.detect_scenes(video, show_progress=False)
        raw = manager.get_scene_list()
        scenes = [(s[0].get_frames(), s[1].get_frames() - 1) for s in raw]
        logger.info(f"      [detect_scenes_simple] {len(scenes)} scenes: {scenes}")
        return scenes
    except Exception as e:
        logger.error(f"      [detect_scenes_simple] error: {e}")
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return [(0, frames - 1)]
            else:
                logger.error("      [detect_scenes_simple] couldn't open video with OpenCV")
                return [(0, 1000)]
        except Exception as ex:
            logger.error(f"      [detect_scenes_simple] fallback error: {ex}")
            return [(0, 1000)]

def normalize_video_fps(input_path: Path, output_path: Path) -> bool:
    try:
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-r", str(FPS_OUT),
            "-c:v", SELECTED_CODEC, "-q:v", "2" if SELECTED_CODEC == "mpeg4" else "23",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        logger.info(f"[normalize_fps] {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0:
            logger.warning(f"[normalize_fps] failed with code {proc.returncode}, stderr: {proc.stderr.decode()}")
            cmd_simple = [
                FFMPEG_BIN, "-y", "-i", str(input_path),
                "-r", str(FPS_OUT), "-c:v", "mpeg4", "-q:v", "2",
                str(output_path)
            ]
            logger.info(f"[normalize_fps] trying simpler command: {' '.join(cmd_simple)}")
            proc_simple = subprocess.run(cmd_simple, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if proc_simple.returncode != 0:
                logger.warning(f"[normalize_fps] simpler command failed with code {proc_simple.returncode}")
                return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 10000
    except Exception as e:
        logger.error(f"[normalize_fps] error: {e}")
        return False

def smart_chunk_segmentation(start: int, end: int) -> List[Tuple[int, int]]:
    length = end - start + 1
    if length < MIN_SEGMENT_LENGTH:
        logger.info(f"[smart_chunk] segment too short ({length} < {MIN_SEGMENT_LENGTH}), extending by {MIN_SEGMENT_LENGTH - length}")
        return [(start, end)]
    if length <= MAX_SEGMENT_LENGTH:
        return [(start, end)]
    num_chunks = max(1, round(length / TARGET_BUCKET))
    chunk_size = max(int(MIN_SEGMENT_LENGTH), length // num_chunks)
    chunks = []
    for i in range(num_chunks):
        chunk_start = start + i * chunk_size
        chunk_end = min(end, chunk_start + chunk_size - 1)
        if i == num_chunks - 1:
            if chunks and chunk_end - chunk_start + 1 < MIN_SEGMENT_LENGTH:
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, chunk_end))
            else:
                chunks.append((chunk_start, chunk_end))
        else:
            chunks.append((chunk_start, chunk_end))
        if chunk_end >= end:
            break
    result = []
    current = None
    for chunk in chunks:
        chunk_len = chunk[1] - chunk[0] + 1
        if chunk_len < MIN_SEGMENT_LENGTH:
            if current:
                current = (current[0], chunk[1])
            else:
                current = chunk
        else:
            if current:
                result.append(current)
                current = None
            result.append(chunk)
    if current:
        result.append(current)
    return result

def detect_scenes_on_range(path: str, s: int, e: int, fps: float) -> List[Tuple[int, int]]:
    start_s = s / fps
    duration = (e - s + 1) / fps
    logger.info(f"      [detect_on_range] extracting {s}-{e} ({duration:.2f}s)")
    length = e - s + 1
    if MIN_SEGMENT_LENGTH <= length <= MAX_SEGMENT_LENGTH:
        logger.info("        [detect_on_range] segment already appropriate length")
        return [(s, e)]
    if length < MIN_SEGMENT_LENGTH * 0.5:
        logger.info("        [detect_on_range] segment too short, returning as is")
        return [(s, e)]
    if length > 1.5 * TARGET_BUCKET:
        chunks = smart_chunk_segmentation(s, e)
        logger.info(f"        [detect_on_range] smart chunking: {chunks}")
        return chunks
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-i", path,
            "-t", f"{duration:.3f}",
            "-c:v", SELECTED_CODEC, "-q:v", "2" if SELECTED_CODEC == "mpeg4" else "23",
            tmp_path
        ]
        logger.info(f"        [detect_on_range] ffmpeg cmd: {' '.join(cmd)}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
            logger.warning("        [detect_on_range] ffmpeg failed or generated empty file")
            return smart_chunk_segmentation(s, e)
        subs = detect_scenes_simple(tmp_path)
        segments = [(a + s, b + s) for (a, b) in subs]
        if not segments or segments == [(s, e)]:
            logger.info("        [detect_on_range] scene detection failed, using smart chunks")
            return smart_chunk_segmentation(s, e)
        logger.info(f"        [detect_on_range] sub-scenes: {segments}")
        fixed_segments = []
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len > MAX_SEGMENT_LENGTH:
                fixed_segments.extend(smart_chunk_segmentation(seg_start, seg_end))
            elif seg_len < MIN_SEGMENT_LENGTH:
                if seg_len >= MIN_SEGMENT_LENGTH * 0.5:
                    fixed_segments.append((seg_start, seg_end))
            else:
                fixed_segments.append((seg_start, seg_end))
        if fixed_segments:
            merged_segments = [fixed_segments[0]]
            for i in range(1, len(fixed_segments)):
                curr_start, curr_end = fixed_segments[i]
                curr_len = curr_end - curr_start + 1
                if curr_len < MIN_SEGMENT_LENGTH * 0.75:
                    prev_start, prev_end = merged_segments[-1]
                    merged_segments[-1] = (prev_start, curr_end)
                else:
                    merged_segments.append((curr_start, curr_end))
            logger.info(f"        [detect_on_range] after merging short segments: {merged_segments}")
            return merged_segments
        return segments
    except Exception as ex:
        logger.error(f"        [detect_on_range] error: {ex}")
        return smart_chunk_segmentation(s, e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception as ex:
            logger.warning(f"        [detect_on_range] could not delete temp: {ex}")

def recursive_split(path: Path, s: int, e: int, fps: float, processed_segments: Set[Tuple[int, int]] = None, depth: int = 0) -> List[Tuple[int, int]]:
    if processed_segments is None:
        processed_segments = set()
    segment_id = (s, e)
    if segment_id in processed_segments or depth >= MAX_RECURSION_DEPTH:
        logger.info(f"    [recursive_split] stopping recursion: depth={depth}, segment={segment_id}")
        if e - s + 1 > MAX_SEGMENT_LENGTH:
            logger.info(f"    [recursive_split] segment still too long, using smart chunks")
            return smart_chunk_segmentation(s, e)
        return [segment_id]
    processed_segments.add(segment_id)
    length = e - s + 1
    logger.info(f"    [recursive_split] {s}-{e}, length={length}, depth={depth}")
    if MIN_SEGMENT_LENGTH <= length <= MAX_SEGMENT_LENGTH:
        logger.info(f"      [recursive_split] within threshold, accept")
        return [(s, e)]
    if length < MIN_SEGMENT_LENGTH * 0.5:
        logger.info(f"      [recursive_split] segment very short, accept as is")
        return [(s, e)]
    subs = detect_scenes_on_range(str(path), s, e, fps)
    is_same = subs == [(s, e)]
    if is_same or not subs:
        logger.info("      [recursive_split] no further splits, using smart chunks")
        return smart_chunk_segmentation(s, e)
    result: List[Tuple[int, int]] = []
    for a, b in subs:
        sub_len = b - a + 1
        if (MIN_SEGMENT_LENGTH <= sub_len <= MAX_SEGMENT_LENGTH) or depth >= MAX_RECURSION_DEPTH - 1:
            logger.info(f"      [recursive_split] sub {a}-{b} accept")
            result.append((a, b))
        elif sub_len > MAX_SEGMENT_LENGTH:
            if (a, b) != (s, e) and (a, b) not in processed_segments and depth < MAX_RECURSION_DEPTH - 1:
                result.extend(recursive_split(path, a, b, fps, processed_segments, depth + 1))
            else:
                logger.info(f"      [recursive_split] avoiding repeated segment {a}-{b}, using smart chunks")
                result.extend(smart_chunk_segmentation(a, b))
        else:
            if sub_len >= MIN_SEGMENT_LENGTH * 0.5:
                result.append((a, b))
            else:
                logger.info(f"      [recursive_split] discarding very short segment {a}-{b}")
    if len(result) >= 2:
        merged = [result[0]]
        for i in range(1, len(result)):
            curr_start, curr_end = result[i]
            curr_len = curr_end - curr_start + 1
            if curr_len < MIN_SEGMENT_LENGTH * 0.75:
                prev_start, prev_end = merged[-1]
                combined_len = curr_end - prev_start + 1
                if combined_len <= MAX_SEGMENT_LENGTH * 1.1:
                    merged[-1] = (prev_start, curr_end)
                else:
                    merged.append((curr_start, curr_end))
            else:
                merged.append((curr_start, curr_end))
        logger.info(f"      [recursive_split] after merging: {len(result)} -> {len(merged)}")
        result = merged
    return result

def ffmpeg_extract(src: Path, dst: Path, start_s: float, duration_s: float) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-i", str(src),
            "-t", f"{duration_s:.3f}", "-r", str(FPS_OUT),
            "-c:v", SELECTED_CODEC, "-q:v", "2" if SELECTED_CODEC == "mpeg4" else "23",
            "-pix_fmt", "yuv420p",
            tmp_path
        ]
        logger.info(f"[ffmpeg_extract] {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if proc.returncode != 0:
            logger.warning(f"[ffmpeg_extract] first attempt failed with code {proc.returncode}, stderr: {proc.stderr.decode()}")
            cmd2 = [
                FFMPEG_BIN, "-y", "-i", str(src),
                "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
                "-r", str(FPS_OUT), "-c:v", "mpeg4", "-q:v", "2",
                tmp_path
            ]
            logger.info(f"[ffmpeg_extract] second attempt: {' '.join(cmd2)}")
            proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if proc2.returncode != 0:
                logger.warning(f"[ffmpeg_extract] second attempt failed with code {proc2.returncode}")
                return False
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
            shutil.move(tmp_path, dst)
            return True
        else:
            logger.warning(f"[ffmpeg_extract] temp file invalid: exists={os.path.exists(tmp_path)}, size={os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0}")
            return False
    except Exception as e:
        logger.error(f"[ffmpeg_extract] exception: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def split_video_into_scenes(video_path: Path) -> Path:
    logger.info(f"[split_video_into_scenes] {video_path.name}")
    out_dir = video_path.parent / f"{video_path.stem}_scenes"
    ensure_dir(out_dir)
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames",
            "-of", "csv=p=0", str(video_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = proc.stdout.strip().split(',')
        if len(info) >= 2:
            fps_parts = info[0].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            total = int(info[1])
        else:
            logger.warning(f"[split_video] ffprobe failed, falling back to estimate")
            fps = FPS_OUT
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(video_path)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            duration = float(proc.stdout.strip()) if proc.returncode == 0 else 0
            total = int(duration * fps) if duration else 1000
        raw = detect_scenes_simple(str(video_path))
        if not raw:
            logger.warning(f"[split_video] no scenes detected, using whole video")
            segments = [(0, total - 1)]
        else:
            starts = [s for s, _ in raw]
            refined = []
            for i, s in enumerate(starts):
                e = starts[i + 1] - 1 if i + 1 < len(starts) else total - 1
                refined.append((s, e))
            segments = []
            for s, e in refined:
                segments.extend(recursive_split(video_path, s, e, fps))
        final_segments = []
        for s, e in segments:
            segment_len = e - s + 1
            if segment_len < MIN_SEGMENT_LENGTH:
                logger.warning(f"[split_video] segment {s}-{e} length {segment_len} < {MIN_SEGMENT_LENGTH}, adjusting")
                extension_needed = MIN_SEGMENT_LENGTH - segment_len
                new_e = min(total - 1, e + extension_needed)
                final_segments.append((s, new_e))
            else:
                final_segments.append((s, e))
        if len(final_segments) != len(segments):
            logger.info(f"[split_video] adjusted segments to meet minimum length: {final_segments}")
            segments = final_segments
        for idx, (s, e) in enumerate(segments):
            out_file = out_dir / f"{video_path.stem}_scene_{idx:04d}.mp4"
            dur = (e - s + 1) / fps
            logger.info(f"[export] {idx}: {s}-{e} ({dur:.2f}s)")
            if out_file.exists() and out_file.stat().st_size > 10000:
                logger.info(f"[export] file already exists, skipping: {out_file}")
                continue
            if not ffmpeg_extract(video_path, out_file, s / fps, dur):
                logger.error(f"[export] ffmpeg failed for segment {s}-{e}")
                continue
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(out_file)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not proc.stdout.strip():
                logger.warning(f"[export] error verifying {out_file}, deleting")
                out_file.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"ERROR in split_video_into_scenes: {e}")
    return out_dir

def has_text(video: Path) -> bool:
    if not OCR_ENABLED:
        logger.info(f"[has_text] OCR disabled: {video.name}")
        return False
    logger.info(f"[has_text] scanning {video.name}")
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            logger.warning(f"  [has_text] cannot open video: {video}")
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total // 10)
        frame_indices = range(0, total, sample_interval)
        frames_with_text = 0
        for frame_idx in tqdm(frame_indices, desc=f"OCR {video.name}"):
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
                        logger.info(f"  [has_text] text found at frame {frame_idx}: {len(confident_texts)} regions")
                        if frames_with_text >= MIN_TEXT_FRAMES:
                            logger.info(f"  [has_text] found {frames_with_text} frames with text, video has subtitles")
                            cap.release()
                            return True
                except Exception as e:
                    logger.warning(f"  [has_text] OCR error on frame {frame_idx}: {e}")
                    continue
        cap.release()
        logger.info(f"  [has_text] found only {frames_with_text} frames with text, insufficient")
        return False
    except Exception as e:
        logger.error(f"ERROR in has_text: {e}")
        return False

def select_bucket(total: int, buckets: List[int]) -> int:
    cands = [b for b in buckets if b <= total]
    return max(cands) if cands else total

def evenly_spaced_indices(total: int, n: int) -> List[int]:
    if n <= 1:
        return [0]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]

def resize_frame(frame: np.ndarray, target_pixels: int = RESOLUTION) -> np.ndarray:
    h, w = frame.shape[:2]
    ratio = w / h
    new_h = int(round(((target_pixels / ratio) ** 0.5) / 16) * 16)
    new_w = int(round((new_h * ratio) / 16) * 16)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def process_video(video: Path, out: Path):
    logger.info(f"[process_video] {video.name} -> {out}")
    try:
        if has_text(video):
            logger.info(f"  [process_video] skipping video with text: {video.name}")
            return
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            logger.warning(f"  [process_video] cannot open video: {video}")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logger.warning(f"  [process_video] invalid frame count: {total}")
            cap.release()
            return
        # Strictly enforce BUCKETS[0] frames (121 in this case)
        required_frames = BUCKETS[0]
        if total < required_frames:
            logger.info(f"  [process_video] skipping video with insufficient frames: {total} < {required_frames}")
            cap.release()
            return
        idxs = set(evenly_spaced_indices(total, required_frames))
        ensure_dir(out.parent)
        vw = None
        processed_frames = 0
        for idx in tqdm(range(total), desc=f"Sample {video.name}"):
            ret, fr = cap.read()
            if not ret:
                break
            if idx in idxs:
                r = resize_frame(fr)
                if vw is None:
                    h, w = r.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"avc1") if SELECTED_CODEC == "libx264" else cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(str(out), fourcc, FPS_OUT, (w, h))
                    if not vw.isOpened():
                        logger.warning(f"  [process_video] avc1 failed, falling back to mp4v")
                        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (w, h))
                vw.write(r)
                processed_frames += 1
        cap.release()
        if vw:
            vw.release()
            # Verify the output has exactly the required number of frames
            check_cap = cv2.VideoCapture(str(out))
            if not check_cap.isOpened() or int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT)) != required_frames:
                logger.warning(f"  [process_video] output file has incorrect frame count or failed, removing: {out}")
                out.unlink(missing_ok=True)
            else:
                logger.info(f"  [process_video] saved {processed_frames} frames to {out}")
            check_cap.release()
    except Exception as e:
        logger.error(f"ERROR in process_video: {e}")
        if out.exists():
            out.unlink(missing_ok=True)

def stage1():
    logger.info("[stage1] start scene detection")
    try:
        video_files = [p for p in INPUT_DIR.rglob('*')
                       if p.suffix.lower() in VIDEO_EXTS and "_scenes" not in str(p)]
        logger.info(f"[stage1] found {len(video_files)} videos to process")
        temp_dir = INPUT_DIR / "normalized_videos"
        ensure_dir(temp_dir)
        normalized_videos = []
        for p in tqdm(video_files, desc="Normalize FPS"):
            norm_path = temp_dir / p.name
            if not norm_path.exists() or norm_path.stat().st_size < 10000:
                success = normalize_video_fps(p, norm_path)
                if success:
                    normalized_videos.append(norm_path)
                else:
                    logger.warning(f"[stage1] failed to normalize {p.name}, using original")
                    normalized_videos.append(p)
            else:
                logger.info(f"[stage1] using existing normalized video {norm_path}")
                normalized_videos.append(norm_path)
        for p in tqdm(normalized_videos, desc="Detect"):
            split_video_into_scenes(p)
    except Exception as e:
        logger.error(f"ERROR in stage1: {e}")
    logger.info("[stage1] done")

def stage2():
    logger.info("[stage2] start OCR & processing")
    try:
        scene_files = [p for p in INPUT_DIR.rglob('*')
                       if p.suffix.lower() in VIDEO_EXTS and "_scene_" in p.name]
        logger.info(f"[stage2] found {len(scene_files)} scene files to process")
        for p in tqdm(scene_files, desc="Process"):
            process_video(p, OUTPUT_DIR / p.relative_to(INPUT_DIR))
    except Exception as e:
        logger.error(f"ERROR in stage2: {e}")
    logger.info("[stage2] done")

if __name__ == "__main__":
    if not INPUT_DIR.exists():
        logger.error("ERROR: INPUT_DIR does not exist:", INPUT_DIR)
        sys.exit(1)
    ensure_dir(OUTPUT_DIR)
    try:
        if "--detect-only" in sys.argv:
            stage1()
        elif "--process-only" in sys.argv:
            stage2()
        else:
            stage1() 
            stage2() 
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}") 
        sys.exit(1)