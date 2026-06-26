"""
Shared utilities for the hybrid YOLOv9-DETR pipeline.

Deduplicates dataset classes, collate functions, the DETR Lightning module,
IoU calculation, and analysis helpers that were repeated across scripts.
"""

import os

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import COCO_ANNOTATION_FILE


# ── Device ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── YOLO-format dataset (used by train_yolo / finetune_yolo) ────────────

class YoloDataset(Dataset):
    def __init__(self, root, transform=None, image_size=(640, 640)):
        self.root = root
        self.transform = transform
        self.image_size = image_size

        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")

        filenames = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
        self.image_paths = [os.path.join(images_dir, f) for f in filenames]
        self.label_paths = [
            os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt") for f in filenames
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        label_path = self.label_paths[idx]
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                values = (values + [0.0] * 5)[:5]
                labels.append(values)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, labels


def yolo_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, 0), list(labels)


# ── COCO-format dataset (used by DETR scripts) ─────────────────────────

class CocoDetection(torchvision.datasets.CocoDetection):
    """Wraps torchvision CocoDetection with DETR processor encoding."""

    def __init__(self, img_folder, processor, return_raw=False):
        ann_file = os.path.join(img_folder, COCO_ANNOTATION_FILE)
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.return_raw = return_raw

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        if self.return_raw:
            return img, target
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


def detr_collate_fn(processor):
    def _collate(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }
    return _collate


# ── DETR Lightning module ──────────────────────────────────────────────

class Detr(pl.LightningModule):
    def __init__(self, model_name_or_path, lr, lr_backbone, weight_decay,
                 scheduler_step_size=None, scheduler_gamma=0.1):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            model_name_or_path,
            num_labels=13,
            ignore_mismatched_sizes=True,
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        for label in labels:
            label["class_labels"] = torch.clamp(
                label["class_labels"], min=0, max=self.model.config.num_labels - 2
            )
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters()
                           if "backbone" not in n and p.requires_grad],
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma,
            )
            return [optimizer], [scheduler]
        return optimizer


# ── IoU ─────────────────────────────────────────────────────────────────

def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = float(area_a + area_b - inter)
    return inter / denom if denom > 0 else 0.0


# ── Analysis helpers ────────────────────────────────────────────────────

def print_final_metrics(trainer):
    training_loss = trainer.callback_metrics.get("training_loss")
    validation_loss = trainer.callback_metrics.get("validation_loss")
    if training_loss is not None and validation_loss is not None:
        print(f"Final Training Loss: {training_loss:.4f}")
        print(f"Final Validation Loss: {validation_loss:.4f}")
    else:
        print("Metrics not found. Ensure the model has been properly trained.")


def plot_confusion_matrix(model_path, val_dataloader):
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.eval()
    model.to(DEVICE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = [t["class_labels"].tolist() for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(model.config.num_labels)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(model.config.id2label.values()),
    )
    disp.plot()
    plt.show()
