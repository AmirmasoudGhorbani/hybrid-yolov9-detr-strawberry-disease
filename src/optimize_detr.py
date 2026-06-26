"""
Stage 4 — Fine-tune detr_strawberry_model_2 (from stage 3) with lower learning rates
and a StepLR scheduler, saving detr_strawberry_model_finetuned. Reports final losses
and plots a confusion matrix.
"""

import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor

from src.config import (
    DETR_BASE_MODEL, DETR_TRAIN_PATH, DETR_VAL_PATH,
    DETR_MODEL_STAGE3, DETR_MODEL_FINETUNED, DETR_PROCESSOR_FINETUNED,
)
from src.utils import CocoDetection, detr_collate_fn, Detr, print_final_metrics, plot_confusion_matrix

processor = DetrImageProcessor.from_pretrained(DETR_BASE_MODEL)

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    torch.set_float32_matmul_precision("medium")

    train_dataset = CocoDetection(img_folder=DETR_TRAIN_PATH, processor=processor)
    val_dataset = CocoDetection(img_folder=DETR_VAL_PATH, processor=processor)
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    collate = detr_collate_fn(processor)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=4, shuffle=True, num_workers=11)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate, batch_size=4, num_workers=11)

    model = Detr(
        model_name_or_path=DETR_MODEL_STAGE3,
        lr=1e-5, lr_backbone=1e-6, weight_decay=1e-4,
        scheduler_step_size=10,
    )
    trainer = pl.Trainer(max_epochs=20, devices=1, accelerator="gpu", log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    model.model.save_pretrained(DETR_MODEL_FINETUNED)
    processor.save_pretrained(DETR_PROCESSOR_FINETUNED)

    print_final_metrics(trainer)
    plot_confusion_matrix(DETR_MODEL_FINETUNED, val_dataloader)
