import argparse
from pathlib import Path
from detect import run_detection
from annotate import annotate_video

def parse_args():
    p = argparse.ArgumentParser(description="Apex Legends enemy detection and annotation")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output annotated video path")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path/name")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=100, help="Max detections per frame")
    p.add_argument("--save-dets", default=None, help="Optional: save detections JSON path")
    p.add_argument("--keep-audio", action="store_true", help="Keep original audio in output")
    return p.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    dets_path = Path(args.save_dets) if args.save_dets else None

    detections = run_detection(
        input_path=input_path,
        model_name=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save_json=dets_path,
    )
    annotate_video(
        input_path=input_path,
        output_path=output_path,
        detections=detections,
        keep_audio=args.keep_audio,
    )

if __name__ == "__main__":
    main()
