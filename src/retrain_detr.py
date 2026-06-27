"""
Improved DETR retraining — single script that replaces stages 3-5.

Key improvements over the original pipeline:
  - Proper id2label/label2id mapping (fixes generic LABEL_0, LABEL_1, etc.)
  - Data augmentation (horizontal flip, colour jitter, sharpness)
  - More epochs (150 total with cosine annealing LR)
  - Gradient clipping for training stability
  - Saves best model based on validation loss
  - Periodic evaluation logging

Run in Google Colab:
    python -m src.retrain_detr
"""

import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor

from src.config import (
    DETR_BASE_MODEL, DETR_TRAIN_PATH, DETR_VAL_PATH,
    DETR_MODEL_V2, DETR_PROCESSOR_V2, ID2LABEL, LABEL2ID,
)
from src.utils import (
    AugmentedCocoDetection, CocoDetection,
    detr_collate_fn, Detr, print_final_metrics,
)


class DetrWithCosineSchedule(Detr):
    def __init__(self, model_name_or_path, lr, lr_backbone, weight_decay,
                 warmup_epochs=10, max_epochs=150):
        super().__init__(
            model_name_or_path=model_name_or_path,
            lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay,
        )
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-7,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self.warmup_epochs,
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[self.warmup_epochs],
        )
        return [optimizer], [{"scheduler": combined, "interval": "epoch"}]


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    torch.set_float32_matmul_precision("medium")

    MAX_EPOCHS = 150
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    processor = DetrImageProcessor.from_pretrained(DETR_BASE_MODEL)

    # Augmented training set, standard validation set
    train_dataset = AugmentedCocoDetection(
        img_folder=DETR_TRAIN_PATH, processor=processor, augment=True,
    )
    val_dataset = CocoDetection(
        img_folder=DETR_VAL_PATH, processor=processor,
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    collate = detr_collate_fn(processor)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = DetrWithCosineSchedule(
        model_name_or_path=DETR_BASE_MODEL,
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
        warmup_epochs=10, max_epochs=MAX_EPOCHS,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="validation_loss", mode="min", save_top_k=1,
        filename="detr-best-{epoch:02d}-{validation_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator="gpu",
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[checkpoint_cb, lr_monitor],
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    print_final_metrics(trainer)

    # Save the best model with proper label mapping
    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        best_model = DetrWithCosineSchedule.load_from_checkpoint(
            best_path,
            model_name_or_path=DETR_BASE_MODEL,
            lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
        )
        best_model.model.config.id2label = ID2LABEL
        best_model.model.config.label2id = LABEL2ID
        best_model.model.save_pretrained(DETR_MODEL_V2)
    else:
        model.model.config.id2label = ID2LABEL
        model.model.config.label2id = LABEL2ID
        model.model.save_pretrained(DETR_MODEL_V2)

    processor.save_pretrained(DETR_PROCESSOR_V2)
    print(f"Model saved to {DETR_MODEL_V2}")
    print(f"Processor saved to {DETR_PROCESSOR_V2}")
