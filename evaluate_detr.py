"""
evaluate_detr.py — Stage 6 of the pipeline.

Evaluates the extra-fine-tuned DETR (`detr_strawberry_model_extrafinetuned`) on the test
set: visualises predictions on a handful of random images, then computes mAP at several
IoU thresholds plus weighted Precision / Recall / F1 and a confusion matrix.

Paths below default to Google Colab / Google Drive — edit them for your environment.

NOTE: the original script relied on `tqdm`, `numpy` and a `calculate_iou` helper without
importing/defining them — they are now included here so the script runs standalone.
"""

import os
import random
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the extra-fine-tuned DETR model (from finetune_detr_extra.py)
detr_model = DetrForObjectDetection.from_pretrained("/content/drive/MyDrive/detr_strawberry_model_extrafinetuned").to(device)
detr_model.eval()  # Set the model to evaluation mode

# Setup the processor for DETR
processor = DetrImageProcessor.from_pretrained("/content/drive/MyDrive/detr_strawberry_processor_extrafinetuned")


# Custom dataset class for COCO format
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
        ann_file = os.path.join(img_folder, '_annotations.coco.json')
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        return img, target


# Load test dataset (same structure as validation)
test_path = '/content/drive/MyDrive/dataset/dataset json format/test'
test_dataset = CocoDetection(img_folder=test_path, processor=processor)


# IoU calculation (was missing in the original script)
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


# Function to visualize the results
def visualize_results(image, scores, labels, boxes, title, color):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=2))
        text = f'{label}: {score:.2f}'
        ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title(title)
    plt.axis('off')
    plt.show()


# Function to run DETR inference on a single test image
def run_detr_inference_on_test(idx):
    img, target = test_dataset[idx]
    image = Image.open(os.path.join(test_path, test_dataset.coco.loadImgs(test_dataset.ids[idx])[0]['file_name']))

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs['pixel_values'])

    processed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)[0]
    scores = processed_outputs['scores'].cpu().numpy()
    labels = processed_outputs['labels'].cpu().numpy()
    boxes = processed_outputs['boxes'].cpu().numpy()

    visualize_results(image, scores, labels, boxes, "DETR Predictions", "yellow")


# Function to evaluate the model: mAP at several IoU thresholds + Precision / Recall / F1
def evaluate_detr_model(detr_model, dataset, processor, iou_thresholds=(0.5, 0.75, 0.95), device='cuda'):
    ap_scores = {iou: [] for iou in iou_thresholds}
    all_y_true = []
    all_y_pred = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating DETR on test set"):
        img, target = dataset[idx]
        image = Image.open(os.path.join(test_path, dataset.coco.loadImgs(dataset.ids[idx])[0]['file_name']))

        # Ground-truth boxes (xywh -> xyxy) and labels
        actual_boxes = [ann['bbox'] for ann in target['annotations']]
        actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in actual_boxes]
        actual_labels = [ann['category_id'] for ann in target['annotations']]

        if len(actual_boxes) == 0:
            continue

        # Run inference with DETR
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detr_model(pixel_values=inputs['pixel_values'])

        processed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)[0]
        pred_boxes = processed_outputs['boxes'].cpu().numpy()
        pred_scores = processed_outputs['scores'].cpu().numpy()
        pred_labels = processed_outputs['labels'].cpu().numpy()

        # Collect labels for Precision / Recall / F1 (align lengths defensively)
        if len(actual_labels) == len(pred_labels):
            all_y_true.extend(actual_labels)
            all_y_pred.extend(pred_labels)
        else:
            min_length = min(len(actual_labels), len(pred_labels))
            all_y_true.extend(actual_labels[:min_length])
            all_y_pred.extend(pred_labels[:min_length])

        # Per-image AP at each IoU threshold
        for iou_thresh in iou_thresholds:
            matches = []
            for pred_box in pred_boxes:
                ious = [calculate_iou(pred_box, gt_box) for gt_box in actual_boxes]
                max_iou = max(ious) if ious else 0
                matches.append(1 if max_iou >= iou_thresh else 0)

            if matches:
                precision, recall, _ = precision_recall_curve(matches, pred_scores)
                ap = average_precision_score(matches, pred_scores)
                ap_scores[iou_thresh].append(ap)

    # Mean AP per IoU threshold
    for iou_thresh in iou_thresholds:
        mean_ap = np.mean(ap_scores[iou_thresh]) if ap_scores[iou_thresh] else 0
        print(f"mAP@{iou_thresh}: {mean_ap:.4f}")

    # Overall weighted Precision / Recall / F1
    precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("Confusion Matrix:\n", confusion_matrix(all_y_true, all_y_pred))


if __name__ == "__main__":
    # Visualise DETR predictions on 10 random test images
    for _ in range(10):
        idx = random.randint(0, len(test_dataset) - 1)
        run_detr_inference_on_test(idx)
    print("Inference on test dataset completed. Yellow boxes represent DETR predictions.")

    # Quantitative evaluation
    evaluate_detr_model(detr_model, test_dataset, processor, device=device)
