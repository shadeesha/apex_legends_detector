import json
from pathlib import Path
from ultralytics import YOLO

# Each detection: {"frame": int, "bbox": [x1, y1, x2, y2], "conf": float, "cls": int, "label": str}
def run_detection(input_path: Path, model_name: str, device: str, conf: float, iou: float, max_det: int, save_json: Path | None):
    model = YOLO(model_name)
    results = model.predict(
        source=str(input_path),
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        stream=True,  # efficient iteration
    )

    detections = []
    for frame_idx, res in enumerate(results):
        names = res.names
        frame_num = getattr(res, "frame", frame_idx)
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = names.get(cls_idx, f"id:{cls_idx}")
            detections.append(
                {
                    "frame": int(frame_num),
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf_score,
                    "cls": cls_idx,
                    "label": label,
                }
            )

    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

    return detections
