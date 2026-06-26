"""
Stage 7 — Hybrid model inference. Runs YOLOv9 and DETR on each test image, fuses their
predictions via IoU-based box selection, and visualises YOLO-only / DETR-only / fused
results side by side.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

from src.config import (
    YOLO_FINETUNE_BEST_PT, DETR_MODEL_EXTRAFINETUNED, DETR_PROCESSOR_EXTRAFINETUNED,
    DETR_TEST_PATH, DISEASE_NAMES,
)
from src.utils import CocoDetection, calculate_iou, DEVICE


def run_yolo_inference(yolo_model, image):
    results = yolo_model.predict(image, stream=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    return boxes, labels


def refine_with_detr(detr_model, processor, image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs["pixel_values"])
    result = processor.post_process_object_detection(
        outputs, target_sizes=[(image.height, image.width)], threshold=0.5,
    )[0]
    return result["boxes"].cpu().numpy(), result["labels"].cpu().numpy()


def select_best_boxes(yolo_boxes, yolo_labels, detr_boxes, detr_labels):
    selected_boxes, selected_labels = [], []
    for i, y_box in enumerate(yolo_boxes):
        y_score = y_box[-1]
        match_found = False
        for j, d_box in enumerate(detr_boxes):
            iou = calculate_iou(y_box[:4], d_box[:4])
            if iou >= 0.5 and d_box[-1] > y_score:
                selected_boxes.append(d_box)
                selected_labels.append(detr_labels[j])
                match_found = True
                break
        if not match_found:
            selected_boxes.append(y_box)
            selected_labels.append(yolo_labels[i])
    for j, d_box in enumerate(detr_boxes):
        if not any(calculate_iou(d_box[:4], s_box[:4]) >= 0.5 for s_box in selected_boxes):
            selected_boxes.append(d_box)
            selected_labels.append(detr_labels[j])
    return selected_boxes, selected_labels


def visualize_model_comparisons(image, yolo_boxes, yolo_labels, detr_boxes, detr_labels,
                                selected_boxes, selected_labels, image_id):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Image ID: {image_id}")

    for ax, boxes, labels, title, color in [
        (axes[0], yolo_boxes, yolo_labels, "YOLO Only (Red)", "red"),
        (axes[1], detr_boxes, detr_labels, "DETR Only (Yellow)", "yellow"),
        (axes[2], selected_boxes, selected_labels, "Selected Boxes (Blue)", "blue"),
    ]:
        ax.imshow(image)
        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor=color, fill=False, linewidth=2))
            ax.text(xmin, ymin, DISEASE_NAMES[int(label)], color=color, fontsize=10)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


def evaluate_test_set(test_dataset, yolo_model, detr_model, processor):
    for idx in range(len(test_dataset)):
        img, target = test_dataset[idx]
        image = Image.open(os.path.join(DETR_TEST_PATH, test_dataset.coco.loadImgs(test_dataset.ids[idx])[0]["file_name"]))

        yolo_boxes, yolo_labels = run_yolo_inference(yolo_model, image)
        detr_boxes, detr_labels = refine_with_detr(detr_model, processor, image)
        selected_boxes, selected_labels = select_best_boxes(yolo_boxes, yolo_labels, detr_boxes, detr_labels)

        if idx < 40:
            visualize_model_comparisons(
                image, yolo_boxes, yolo_labels, detr_boxes, detr_labels,
                selected_boxes, selected_labels, test_dataset.ids[idx],
            )


if __name__ == "__main__":
    yolo_model = YOLO(YOLO_FINETUNE_BEST_PT)
    yolo_model.fuse()
    yolo_model.to(DEVICE)
    yolo_model.overrides["mode"] = "predict"

    detr_model = DetrForObjectDetection.from_pretrained(DETR_MODEL_EXTRAFINETUNED).to(DEVICE)
    detr_model.eval()
    processor = DetrImageProcessor.from_pretrained(DETR_PROCESSOR_EXTRAFINETUNED)

    test_dataset = CocoDetection(img_folder=DETR_TEST_PATH, processor=processor, return_raw=True)
    evaluate_test_set(test_dataset, yolo_model, detr_model, processor)
    print("Visualization completed.")
