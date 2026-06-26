"""
Stage 3 — Train DETR (facebook/detr-resnet-50) on the strawberry dataset in COCO format.
Saves detr_strawberry_model_2 and runs a quick sanity-check visualisation.
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor

from src.config import (
    DETR_BASE_MODEL, DETR_TRAIN_PATH, DETR_VAL_PATH,
    DETR_MODEL_STAGE3, DETR_PROCESSOR_STAGE3,
)
from src.utils import CocoDetection, detr_collate_fn, Detr

processor = DetrImageProcessor.from_pretrained(DETR_BASE_MODEL)


def plot_results(pil_img, scores, labels, boxes, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]] * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(
        scores.tolist(), labels.tolist(), boxes.tolist(), colors
    ):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f"{model.model.config.id2label[label]}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


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

    model = Detr(model_name_or_path=DETR_BASE_MODEL, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    trainer = pl.Trainer(max_epochs=30, devices=1, accelerator="gpu", log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    model.model.save_pretrained(DETR_MODEL_STAGE3)
    processor.save_pretrained(DETR_PROCESSOR_STAGE3)

    # Sanity-check inference on a validation image
    image_id = val_dataset.coco.getImgIds()[0]
    image_info = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(DETR_VAL_PATH, image_info["file_name"]))

    pixel_values, _ = val_dataset[0]
    pixel_values = pixel_values.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model.model(pixel_values=pixel_values)

    width, height = image.size
    results = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.9)[0]
    plot_results(image, results["scores"], results["labels"], results["boxes"], model)

    # Visualise a random ground-truth annotation
    image_ids = train_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print(f"Image #{image_id}")

    image_info = train_dataset.coco.loadImgs(image_id)[0]
    annotations = train_dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(train_dataset.root, image_info["file_name"])
    image = cv2.imread(image_path)

    categories = train_dataset.coco.cats
    id2label = {k: v["name"] for k, v in categories.items()}

    plt.figure(figsize=(10, 10))
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        cv2.putText(image, id2label[ann["category_id"]], (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
