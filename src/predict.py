"""
Single-image (or directory) inference CLI for the hybrid YOLOv9-DETR pipeline.

Usage:
    python -m src.predict --image path/to/image.jpg
    python -m src.predict --image path/to/image.jpg --output result.jpg
    python -m src.predict --image-dir path/to/folder/ --output-dir results/
    python -m src.predict --image path/to/image.jpg --threshold 0.4 --iou-threshold 0.6
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

from src.config import (
    YOLO_FINETUNE_BEST_PT, DETR_MODEL_EXTRAFINETUNED,
    DETR_PROCESSOR_EXTRAFINETUNED, DISEASE_NAMES,
)
from src.utils import calculate_iou, DEVICE


def load_models(yolo_weights=None, detr_model_path=None, detr_processor_path=None):
    yolo_weights = yolo_weights or YOLO_FINETUNE_BEST_PT
    detr_model_path = detr_model_path or DETR_MODEL_EXTRAFINETUNED
    detr_processor_path = detr_processor_path or DETR_PROCESSOR_EXTRAFINETUNED

    yolo_model = YOLO(yolo_weights)
    yolo_model.fuse()
    yolo_model.to(DEVICE)
    yolo_model.overrides["mode"] = "predict"

    detr_model = DetrForObjectDetection.from_pretrained(detr_model_path).to(DEVICE)
    detr_model.eval()
    processor = DetrImageProcessor.from_pretrained(detr_processor_path)

    return yolo_model, detr_model, processor


def run_hybrid_inference(image, yolo_model, detr_model, processor,
                         threshold=0.5, iou_threshold=0.5):
    yolo_results = yolo_model.predict(image, stream=False, verbose=False)
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])
    yolo_scores = yolo_results[0].boxes.conf.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])
    yolo_labels = yolo_results[0].boxes.cls.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs["pixel_values"])
    result = processor.post_process_object_detection(
        outputs, target_sizes=[(image.height, image.width)], threshold=threshold,
    )[0]
    detr_boxes = result["boxes"].cpu().numpy()
    detr_scores = result["scores"].cpu().numpy()
    detr_labels = result["labels"].cpu().numpy()

    selected_boxes, selected_labels, selected_scores, selected_sources = [], [], [], []

    for i in range(len(yolo_boxes)):
        match_found = False
        for j in range(len(detr_boxes)):
            iou = calculate_iou(yolo_boxes[i], detr_boxes[j])
            if iou >= iou_threshold and detr_scores[j] > yolo_scores[i]:
                selected_boxes.append(detr_boxes[j])
                selected_labels.append(detr_labels[j])
                selected_scores.append(detr_scores[j])
                selected_sources.append("detr")
                match_found = True
                break
        if not match_found:
            selected_boxes.append(yolo_boxes[i])
            selected_labels.append(yolo_labels[i])
            selected_scores.append(yolo_scores[i])
            selected_sources.append("yolo")

    for j in range(len(detr_boxes)):
        if not selected_boxes or not any(
            calculate_iou(detr_boxes[j], s_box) >= iou_threshold for s_box in selected_boxes
        ):
            selected_boxes.append(detr_boxes[j])
            selected_labels.append(detr_labels[j])
            selected_scores.append(detr_scores[j])
            selected_sources.append("detr")

    return {
        "boxes": selected_boxes,
        "labels": selected_labels,
        "scores": selected_scores,
        "sources": selected_sources,
        "yolo_raw": {"boxes": yolo_boxes, "labels": yolo_labels, "scores": yolo_scores},
        "detr_raw": {"boxes": detr_boxes, "labels": detr_labels, "scores": detr_scores},
    }


def draw_predictions(image, results, show_comparison=True):
    if show_comparison:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle("Hybrid YOLOv9-DETR Strawberry Disease Detection", fontsize=16, fontweight="bold")

        panels = [
            (axes[0], results["yolo_raw"]["boxes"], results["yolo_raw"]["labels"],
             results["yolo_raw"]["scores"], "YOLOv9 Detections", "#e74c3c"),
            (axes[1], results["detr_raw"]["boxes"], results["detr_raw"]["labels"],
             results["detr_raw"]["scores"], "DETR Detections", "#f39c12"),
            (axes[2], results["boxes"], results["labels"],
             results["scores"], "Hybrid (Fused)", "#2ecc71"),
        ]
        for ax, boxes, labels, scores, title, color in panels:
            ax.imshow(image)
            for box, label, score in zip(boxes, labels, scores):
                xmin, ymin, xmax, ymax = box[:4]
                name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           edgecolor=color, fill=False, linewidth=2.5))
                ax.text(xmin, ymin - 4, f"{name} {score:.0%}",
                        color="white", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        source_colors = {"yolo": "#e74c3c", "detr": "#f39c12"}
        for box, label, score, source in zip(
            results["boxes"], results["labels"], results["scores"], results["sources"]
        ):
            xmin, ymin, xmax, ymax = box[:4]
            name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
            color = source_colors.get(source, "#2ecc71")
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       edgecolor=color, fill=False, linewidth=2.5))
            ax.text(xmin, ymin - 4, f"{name} {score:.0%} [{source.upper()}]",
                    color="white", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
        ax.set_title("Hybrid YOLOv9-DETR Detection", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
    return fig


def predict_image(image_path, yolo_model, detr_model, processor,
                  output_path=None, threshold=0.5, iou_threshold=0.5,
                  show_comparison=True):
    image = Image.open(image_path).convert("RGB")
    results = run_hybrid_inference(image, yolo_model, detr_model, processor,
                                  threshold=threshold, iou_threshold=iou_threshold)

    fig = draw_predictions(image, results, show_comparison=show_comparison)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        plt.show()

    n = len(results["boxes"])
    print(f"  {os.path.basename(image_path)}: {n} detection{'s' if n != 1 else ''}")
    for box, label, score, source in zip(
        results["boxes"], results["labels"], results["scores"], results["sources"]
    ):
        name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
        print(f"    - {name}: {score:.1%} (from {source.upper()})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid YOLOv9-DETR inference on strawberry images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output", type=str, help="Output path for single-image result")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch results")
    parser.add_argument("--threshold", type=float, default=0.5, help="DETR confidence threshold (default: 0.5)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for box fusion (default: 0.5)")
    parser.add_argument("--no-comparison", action="store_true", help="Show only fused result, not the 3-panel comparison")
    parser.add_argument("--yolo-weights", type=str, help="Override YOLO weights path")
    parser.add_argument("--detr-model", type=str, help="Override DETR model path")
    parser.add_argument("--detr-processor", type=str, help="Override DETR processor path")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    print("Loading models...")
    yolo_model, detr_model, processor = load_models(
        args.yolo_weights, args.detr_model, args.detr_processor,
    )
    print(f"Models loaded on {DEVICE}")

    show_comparison = not args.no_comparison

    if args.image:
        predict_image(args.image, yolo_model, detr_model, processor,
                      output_path=args.output, threshold=args.threshold,
                      iou_threshold=args.iou_threshold, show_comparison=show_comparison)

    elif args.image_dir:
        output_dir = args.output_dir or os.path.join(args.image_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(
            f for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in extensions
        )
        print(f"Processing {len(image_files)} images...")
        for fname in image_files:
            out_name = os.path.splitext(fname)[0] + "_result.jpg"
            predict_image(
                os.path.join(args.image_dir, fname), yolo_model, detr_model, processor,
                output_path=os.path.join(output_dir, out_name),
                threshold=args.threshold, iou_threshold=args.iou_threshold,
                show_comparison=show_comparison,
            )
        print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
