from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_YAML = BASE_DIR / "data.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=-1,              # auto batch size
    patience=20,           # early stopping
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    cos_lr=True,
    close_mosaic=10,
    project=str(BASE_DIR / "runs" / "detect"),
    name="train",
    exist_ok=True,
    verbose=True,
)