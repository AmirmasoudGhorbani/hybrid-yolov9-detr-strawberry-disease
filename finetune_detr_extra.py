"""
finetune_detr_extra.py — Stage 5 of the pipeline.

A second fine-tuning pass: loads `detr_strawberry_model_finetuned` (from optimize_detr.py)
and trains it further (more epochs), saving `detr_strawberry_model_extrafinetuned` — the
DETR weights used for evaluation (stage 6) and in the hybrid model (stage 7).

Paths below default to Google Colab / Google Drive — edit them for your environment.
"""

import os
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

# Define paths to your dataset
train_path = '/content/drive/MyDrive/dataset/dataset json format/train'
val_path = '/content/drive/MyDrive/dataset/dataset json format/valid'
annotation_file = '_annotations.coco.json'

# Setup DetrImageProcessor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, annotation_file)
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


train_dataset = CocoDetection(img_folder=train_path, processor=processor)
val_dataset = CocoDetection(img_folder=val_path, processor=processor, train=False)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True, num_workers=11)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4, num_workers=11)


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # Load the DETR fine-tuned in stage 4 (optimize_detr.py)
        self.model = DetrForObjectDetection.from_pretrained(
            "/content/drive/MyDrive/detr_strawberry_model_finetuned",
            num_labels=13,  # 12 classes + 1 for 'no object'
            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        for label in labels:
            label["class_labels"] = torch.clamp(label["class_labels"], min=0, max=self.model.config.num_labels - 2)
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
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # StepLR scheduler
        return [optimizer], [lr_scheduler]


def analyze_model_performance(trainer):
    training_loss = trainer.callback_metrics.get('training_loss', None)
    validation_loss = trainer.callback_metrics.get('validation_loss', None)
    if training_loss is not None and validation_loss is not None:
        print(f"Final Training Loss: {training_loss:.4f}")
        print(f"Final Validation Loss: {validation_loss:.4f}")
    else:
        print("Metrics not found. Ensure the model has been properly trained.")


def plot_confusion_matrix(model_path, val_dataloader):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(model.device)
            labels = [t['class_labels'].tolist() for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(model.config.num_labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(model.config.id2label.values()))
    disp.plot()
    plt.show()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_float32_matmul_precision('medium')
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    model = Detr(lr=1e-5, lr_backbone=1e-6, weight_decay=1e-4)  # Lower learning rates for fine-tuning
    trainer = pl.Trainer(max_epochs=50, devices=1, accelerator='gpu', log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the extra-fine-tuned model
    model.model.save_pretrained("/content/drive/MyDrive/detr_strawberry_model_extrafinetuned")
    processor.save_pretrained("/content/drive/MyDrive/detr_strawberry_processor_extrafinetuned")

    analyze_model_performance(trainer)
    plot_confusion_matrix("/content/drive/MyDrive/detr_strawberry_model_extrafinetuned", val_dataloader)
