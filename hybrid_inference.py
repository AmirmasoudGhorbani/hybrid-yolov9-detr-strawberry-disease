"""
hybrid_inference.py — Stage 7 of the pipeline (the hybrid model).

Runs YOLOv9 and DETR on each test image, then fuses their predictions: for every YOLO
box it looks for a higher-confidence DETR box with IoU >= 0.5 and keeps the better one,
adding any DETR-only detections. Visualises YOLO-only, DETR-only and the fused result
side by side.

Inputs:
  - fine-tuned YOLO weights  (from finetune_yolo.py)
  - DETR `..._model_extrafinetuned` + processor (from finetune_detr_extra.py)

Paths below default to Google Colab / Google Drive — edit them for your environment.
"""

import os
import torch
import torchvision
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv9 model (fine-tuned weights from stage 2)
yolo_model = YOLO('/content/drive/MyDrive/yolov9_lastfinetune_results/A100_finetune_best_resume/weights/best.pt')
yolo_model.fuse()
yolo_model.to(device)
yolo_model.overrides['mode'] = 'predict'

# Load DETR model and processor (extra-fine-tuned, from stage 5)
detr_model = DetrForObjectDetection.from_pretrained("/content/drive/MyDrive/detr_strawberry_model_extrafinetuned").to(device)
detr_model.eval()
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


# Load test dataset
test_dataset = CocoDetection(img_folder='/content/drive/MyDrive/dataset/dataset json format/test', processor=processor)


# YOLO inference function
def run_yolo_inference(image):
    results = yolo_model.predict(image, stream=False)
    yolo_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    yolo_labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    return yolo_boxes, yolo_labels


# DETR inference function
def refine_with_detr(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs['pixel_values'])
    processed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)[0]
    detr_boxes = processed_outputs['boxes'].cpu().numpy()
    detr_labels = processed_outputs['labels'].cpu().numpy()
    return detr_boxes, detr_labels


# Function to select best bounding boxes
def select_best_boxes(yolo_boxes, yolo_labels, detr_boxes, detr_labels):
    selected_boxes = []
    selected_labels = []
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


# IoU calculation
def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Disease / ripeness class names (index order matches the trained models)
DISEASE_NAMES = ["Angular Leafspot", "Anthracnose Fruit Rot", "Early-Turning", "Gray Mold", "Green-Strawberry",
                 "Late-Turning", "Leaf Spot", "Powdery Mildew Fruit", "Powdery Mildew Leaf", "Red-Turning",
                 "Turning", "White-Strawberry"]


# Visualization function
def visualize_model_comparisons(image, yolo_boxes, yolo_labels, detr_boxes, detr_labels, selected_boxes, selected_labels, image_id):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Image ID: {image_id}")

    # YOLO Only
    axes[0].imshow(image)
    for (xmin, ymin, xmax, ymax), label in zip(yolo_boxes, yolo_labels):
        disease_name = DISEASE_NAMES[int(label)]
        axes[0].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor="red", fill=False, linewidth=2))
        axes[0].text(xmin, ymin, disease_name, color="red", fontsize=10)
    axes[0].set_title("YOLO Only (Red)")
    axes[0].axis("off")

    # DETR Only
    axes[1].imshow(image)
    for (xmin, ymin, xmax, ymax), label in zip(detr_boxes, detr_labels):
        disease_name = DISEASE_NAMES[int(label)]
        axes[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor="yellow", fill=False, linewidth=2))
        axes[1].text(xmin, ymin, disease_name, color="yellow", fontsize=10)
    axes[1].set_title("DETR Only (Yellow)")
    axes[1].axis("off")

    # Selected Boxes
    axes[2].imshow(image)
    for (xmin, ymin, xmax, ymax), label in zip(selected_boxes, selected_labels):
        disease_name = DISEASE_NAMES[int(label)]
        axes[2].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor="blue", fill=False, linewidth=2))
        axes[2].text(xmin, ymin, disease_name, color="blue", fontsize=10)
    axes[2].set_title("Selected Boxes (Blue)")
    axes[2].axis("off")

    plt.show()


# Evaluation function
def evaluate_test_set(test_dataset):
    for idx in range(len(test_dataset)):
        img, target = test_dataset[idx]
        image = Image.open(os.path.join('/content/drive/MyDrive/dataset/dataset json format/test', test_dataset.coco.loadImgs(test_dataset.ids[idx])[0]['file_name']))

        # YOLO predictions
        yolo_boxes, yolo_labels = run_yolo_inference(image)

        # DETR predictions
        detr_boxes, detr_labels = refine_with_detr(image)

        # Select best bounding boxes
        selected_boxes, selected_labels = select_best_boxes(yolo_boxes, yolo_labels, detr_boxes, detr_labels)

        # Visualize for the first few images
        if idx < 40:
            visualize_model_comparisons(image, yolo_boxes, yolo_labels, detr_boxes, detr_labels, selected_boxes, selected_labels, test_dataset.ids[idx])


if __name__ == "__main__":
    # Run evaluation on the entire test set
    evaluate_test_set(test_dataset)
    print("Visualization completed.")
