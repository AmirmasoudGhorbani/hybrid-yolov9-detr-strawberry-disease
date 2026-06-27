"""
Centralised path configuration for the hybrid YOLOv9-DETR pipeline.

Edit the paths below to match your environment. The defaults point at
Google Drive locations used during the original Colab development.
"""

import os

# ── Dataset paths (COCO JSON format, used by DETR scripts) ──────────────
DETR_TRAIN_PATH = "/content/drive/MyDrive/dataset/dataset json format/train"
DETR_VAL_PATH = "/content/drive/MyDrive/dataset/dataset json format/valid"
DETR_TEST_PATH = "/content/drive/MyDrive/dataset/dataset json format/test"
COCO_ANNOTATION_FILE = "_annotations.coco.json"

# ── Dataset paths (YOLO format, used by YOLO scripts) ───────────────────
YOLO_TRAIN_ROOT = "/content/drive/MyDrive/yolov9 dataset/train"
YOLO_VAL_ROOT = "/content/drive/MyDrive/yolov9 dataset/val"
YOLO_DATA_YAML = "/content/drive/MyDrive/yolov9 dataset/data.yaml"

# ── YOLO weight / output paths ──────────────────────────────────────────
YOLO_TRAIN_PROJECT = "/content/drive/MyDrive/yolov9_Training"
YOLO_TRAIN_NAME = "A100_experiment"
YOLO_BEST_PT = os.path.join(YOLO_TRAIN_PROJECT, YOLO_TRAIN_NAME, "weights/best.pt")

YOLO_FINETUNE_PROJECT = "/content/drive/MyDrive/yolov9_lastfinetune_results"
YOLO_FINETUNE_NAME = "A100_finetune_best_resume"
YOLO_FINETUNE_BEST_PT = os.path.join(YOLO_FINETUNE_PROJECT, YOLO_FINETUNE_NAME, "weights/best.pt")

# ── DETR weight / output paths ──────────────────────────────────────────
DETR_BASE_MODEL = "facebook/detr-resnet-50"
DETR_MODEL_STAGE3 = "/content/drive/MyDrive/detr_strawberry_model_2"
DETR_PROCESSOR_STAGE3 = "/content/drive/MyDrive/detr_strawberry_processor_2"

DETR_MODEL_FINETUNED = "/content/drive/MyDrive/detr_strawberry_model_finetuned"
DETR_PROCESSOR_FINETUNED = "/content/drive/MyDrive/detr_strawberry_processor_finetuned"

DETR_MODEL_EXTRAFINETUNED = "/content/drive/MyDrive/detr_strawberry_model_extrafinetuned"
DETR_PROCESSOR_EXTRAFINETUNED = "/content/drive/MyDrive/detr_strawberry_processor_extrafinetuned"

# ── Improved DETR output paths (v2 retraining) ─────────────────────────
DETR_MODEL_V2 = "/content/drive/MyDrive/detr_strawberry_v2"
DETR_PROCESSOR_V2 = "/content/drive/MyDrive/detr_strawberry_processor_v2"

# ── Model constants ─────────────────────────────────────────────────────
NUM_LABELS = 13  # 12 disease/ripeness classes + 1 background

DISEASE_NAMES = [
    "Angular Leafspot",
    "Anthracnose Fruit Rot",
    "Early-Turning",
    "Gray Mold",
    "Green-Strawberry",
    "Late-Turning",
    "Leaf Spot",
    "Powdery Mildew Fruit",
    "Powdery Mildew Leaf",
    "Red-Turning",
    "Turning",
    "White-Strawberry",
]

# Proper id2label / label2id for DETR config (was missing in original training)
ID2LABEL = {i: name for i, name in enumerate(DISEASE_NAMES)}
ID2LABEL[12] = "no-object"
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
