import subprocess
import tempfile
import json
from pathlib import Path
import cv2
import os

BOX_COLOR = (0, 0, 255)  # BGR red
TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

def annotate_video(input_path: Path, output_path: Path, detections, keep_audio: bool):
    # Build map: frame -> list of boxes
    frame_to_dets = {}
    for det in detections:
        frame_to_dets.setdefault(det["frame"], []).append(det)

    # Read video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write annotated temp video (no audio)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = frame_to_dets.get(frame_idx, [])
        for det in dets:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['label']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()

    ffmpeg_bin = os.environ.get("FFMPEG_BIN", r"C:\Program Files (x86)\ffmpeg-8.0.1\bin\ffmpeg.exe")

    if keep_audio:
        # Remux audio from source to annotated video
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(tmp_out),
            "-i", str(input_path),
            "-c", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        output_path.write_bytes(tmp_out.read_bytes())

    tmp_out.unlink(missing_ok=True)
