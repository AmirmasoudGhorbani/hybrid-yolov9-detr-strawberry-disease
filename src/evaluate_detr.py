"""
Stage 6 — Evaluate the extra-fine-tuned DETR on the test set: visualise predictions on
random images, then compute mAP at several IoU thresholds plus weighted P/R/F1 and a
confusion matrix.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection

from src.config import DETR_TEST_PATH, DETR_MODEL_EXTRAFINETUNED, DETR_PROCESSOR_EXTRAFINETUNED
from src.utils import CocoDetection, calculate_iou, DEVICE


def visualize_results(image, scores, labels, boxes, title, color):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=2))
        ax.text(xmin, ymin, f"{label}: {score:.2f}", fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_detr_inference_on_test(idx, test_dataset, detr_model, processor):
    img, target = test_dataset[idx]
    image = Image.open(os.path.join(DETR_TEST_PATH, test_dataset.coco.loadImgs(test_dataset.ids[idx])[0]["file_name"]))

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs["pixel_values"])

    result = processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)[0]
    visualize_results(image, result["scores"].cpu().numpy(), result["labels"].cpu().numpy(),
                      result["boxes"].cpu().numpy(), "DETR Predictions", "yellow")


def evaluate_detr_model(detr_model, dataset, processor, iou_thresholds=(0.5, 0.75, 0.95)):
    ap_scores = {iou: [] for iou in iou_thresholds}
    all_y_true, all_y_pred = [], []

    for idx in tqdm(range(len(dataset)), desc="Evaluating DETR on test set"):
        img, target = dataset[idx]
        image = Image.open(os.path.join(DETR_TEST_PATH, dataset.coco.loadImgs(dataset.ids[idx])[0]["file_name"]))

        actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in (ann["bbox"] for ann in target["annotations"])]
        actual_labels = [ann["category_id"] for ann in target["annotations"]]
        if not actual_boxes:
            continue

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = detr_model(pixel_values=inputs["pixel_values"])

        result = processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)[0]
        pred_boxes = result["boxes"].cpu().numpy()
        pred_scores = result["scores"].cpu().numpy()
        pred_labels = result["labels"].cpu().numpy()

        min_len = min(len(actual_labels), len(pred_labels))
        all_y_true.extend(actual_labels[:min_len])
        all_y_pred.extend(pred_labels[:min_len])

        for iou_thresh in iou_thresholds:
            matches = []
            for pred_box in pred_boxes:
                ious = [calculate_iou(pred_box, gt_box) for gt_box in actual_boxes]
                matches.append(1 if (ious and max(ious) >= iou_thresh) else 0)
            if matches:
                precision_recall_curve(matches, pred_scores)
                ap_scores[iou_thresh].append(average_precision_score(matches, pred_scores))

    for iou_thresh in iou_thresholds:
        mean_ap = np.mean(ap_scores[iou_thresh]) if ap_scores[iou_thresh] else 0
        print(f"mAP@{iou_thresh}: {mean_ap:.4f}")

    print(f"Precision: {precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score: {f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_y_true, all_y_pred))


if __name__ == "__main__":
    detr_model = DetrForObjectDetection.from_pretrained(DETR_MODEL_EXTRAFINETUNED).to(DEVICE)
    detr_model.eval()
    processor = DetrImageProcessor.from_pretrained(DETR_PROCESSOR_EXTRAFINETUNED)
    test_dataset = CocoDetection(img_folder=DETR_TEST_PATH, processor=processor, return_raw=True)

    for _ in range(10):
        idx = random.randint(0, len(test_dataset) - 1)
        run_detr_inference_on_test(idx, test_dataset, detr_model, processor)
    print("Inference on test dataset completed.")

    evaluate_detr_model(detr_model, test_dataset, processor)
