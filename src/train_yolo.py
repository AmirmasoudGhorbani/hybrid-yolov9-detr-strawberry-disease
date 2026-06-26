"""
Stage 1 — Train YOLOv9s from scratch on the strawberry dataset (YOLO format).
Produces the initial best.pt that finetune_yolo.py (stage 2) refines.
"""

from torch.utils.data import DataLoader
from ultralytics import YOLO

from src.config import YOLO_TRAIN_ROOT, YOLO_VAL_ROOT, YOLO_DATA_YAML, YOLO_TRAIN_PROJECT, YOLO_TRAIN_NAME
from src.utils import YoloDataset, yolo_collate_fn

if __name__ == "__main__":
    train_dataset = YoloDataset(root=YOLO_TRAIN_ROOT)
    val_dataset = YoloDataset(root=YOLO_VAL_ROOT)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=yolo_collate_fn)

    model = YOLO("yolov9s.pt")
    model.train(
        data=YOLO_DATA_YAML,
        epochs=20,
        project=YOLO_TRAIN_PROJECT,
        name=YOLO_TRAIN_NAME,
        batch=32,
        workers=4,
        device=0,
        amp=True,
        cos_lr=True,
        mosaic=1,
        mixup=0.2,
        auto_augment="randaugment",
        label_smoothing=0.1,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        imgsz=1024,
        patience=10,
    )
    print("Training completed.")
