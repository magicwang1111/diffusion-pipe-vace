import cv2
from pathlib import Path

root = Path("/home/wangxi/diffusion-pipe/dataset/20250417chi处理后v4已打标")
video_paths = list(root.rglob("*"))
video_paths = [p for p in video_paths if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]]

for path in video_paths:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ar = round(w / h, 2) if h else 0
    print(f"{path.name}: {frame_count} frames, {w}x{h}, AR={ar}")
    cap.release()
