"""
Gradio app for the Hybrid YOLOv9-DETR Strawberry Disease Detector.

Deploy on HuggingFace Spaces or run locally:
    cd space && python app.py
"""

import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

# ── Configuration ───────────────────────────────────────────────────────
# Update this to your HuggingFace Hub repo once you upload the weights
HF_REPO_ID = "Amir-masoud-gh96/hybrid-yolov9-detr-strawberry"

DISEASE_NAMES = [
    "Angular Leafspot", "Anthracnose Fruit Rot", "Early-Turning", "Gray Mold",
    "Green-Strawberry", "Late-Turning", "Leaf Spot", "Powdery Mildew Fruit",
    "Powdery Mildew Leaf", "Red-Turning", "Turning", "White-Strawberry",
]

DISEASE_INFO = {
    "Angular Leafspot": {"type": "Disease", "severity": "Moderate", "description": "Bacterial infection causing angular, water-soaked lesions on leaves."},
    "Anthracnose Fruit Rot": {"type": "Disease", "severity": "High", "description": "Fungal disease causing dark, sunken lesions on fruit. Can cause significant crop loss."},
    "Gray Mold": {"type": "Disease", "severity": "High", "description": "Caused by Botrytis cinerea. Appears as fuzzy gray growth, especially in humid conditions."},
    "Leaf Spot": {"type": "Disease", "severity": "Low-Moderate", "description": "Circular spots on leaves caused by various fungal pathogens."},
    "Powdery Mildew Fruit": {"type": "Disease", "severity": "Moderate", "description": "White powdery coating on fruit surface caused by fungal infection."},
    "Powdery Mildew Leaf": {"type": "Disease", "severity": "Moderate", "description": "White powdery patches on leaf surfaces, curling and distortion."},
    "Early-Turning": {"type": "Ripeness", "severity": "N/A", "description": "Fruit beginning to change colour from green to white/pink."},
    "Late-Turning": {"type": "Ripeness", "severity": "N/A", "description": "Fruit mostly red with some remaining white/pink areas."},
    "Green-Strawberry": {"type": "Ripeness", "severity": "N/A", "description": "Immature, fully green fruit still developing."},
    "Red-Turning": {"type": "Ripeness", "severity": "N/A", "description": "Fruit actively reddening, approaching full ripeness."},
    "Turning": {"type": "Ripeness", "severity": "N/A", "description": "Fruit in mid-transition between green and red stages."},
    "White-Strawberry": {"type": "Ripeness", "severity": "N/A", "description": "Fruit at the white stage before colour development begins."},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Loading ───────────────────────────────────────────────────────

def load_models():
    cache_dir = os.path.join(os.path.dirname(__file__), ".model_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Download all files from the HF repo
    repo_dir = snapshot_download(repo_id=HF_REPO_ID, cache_dir=cache_dir)

    # YOLO weights (flat: best.pt in repo root)
    yolo_path = os.path.join(repo_dir, "best.pt")
    yolo_model = YOLO(yolo_path)
    yolo_model.fuse()
    yolo_model.to(DEVICE)
    yolo_model.overrides["mode"] = "predict"

    # DETR model — config.json and model.safetensors are in the repo root
    detr_model = DetrForObjectDetection.from_pretrained(repo_dir).to(DEVICE)
    detr_model.eval()

    # DETR processor — preprocessor_config.json is in the repo root
    processor = DetrImageProcessor.from_pretrained(repo_dir)

    return yolo_model, detr_model, processor


# ── Inference ───────────────────────────────────────────────────────────

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


def run_inference(image, confidence_threshold, iou_threshold):
    yolo_results = yolo_model.predict(image, stream=False, verbose=False)
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])
    yolo_scores = yolo_results[0].boxes.conf.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])
    yolo_labels = yolo_results[0].boxes.cls.cpu().numpy() if yolo_results[0].boxes is not None else np.array([])

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detr_model(pixel_values=inputs["pixel_values"])
    result = processor.post_process_object_detection(
        outputs, target_sizes=[(image.height, image.width)], threshold=confidence_threshold,
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
                selected_scores.append(float(detr_scores[j]))
                selected_sources.append("DETR")
                match_found = True
                break
        if not match_found:
            selected_boxes.append(yolo_boxes[i])
            selected_labels.append(yolo_labels[i])
            selected_scores.append(float(yolo_scores[i]))
            selected_sources.append("YOLO")

    for j in range(len(detr_boxes)):
        if not selected_boxes or not any(
            calculate_iou(detr_boxes[j], s_box) >= iou_threshold for s_box in selected_boxes
        ):
            selected_boxes.append(detr_boxes[j])
            selected_labels.append(detr_labels[j])
            selected_scores.append(float(detr_scores[j]))
            selected_sources.append("DETR")

    return {
        "boxes": selected_boxes, "labels": selected_labels,
        "scores": selected_scores, "sources": selected_sources,
        "yolo": {"boxes": yolo_boxes, "labels": yolo_labels, "scores": yolo_scores},
        "detr": {"boxes": detr_boxes, "labels": detr_labels, "scores": detr_scores},
    }


# ── Visualisation ───────────────────────────────────────────────────────

def draw_panel(ax, image, boxes, labels, scores, title, color):
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box[:4]
        name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            edgecolor=color, fill=False, linewidth=2.5,
        ))
        ax.text(
            xmin, ymin - 4, f"{name} {score:.0%}",
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")


def create_comparison_figure(image, results):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Hybrid YOLOv9-DETR Strawberry Disease Detection",
        fontsize=16, fontweight="bold", y=0.98,
    )
    draw_panel(axes[0], image, results["yolo"]["boxes"], results["yolo"]["labels"],
               results["yolo"]["scores"], "YOLOv9 Detections", "#e74c3c")
    draw_panel(axes[1], image, results["detr"]["boxes"], results["detr"]["labels"],
               results["detr"]["scores"], "DETR Detections", "#f39c12")
    draw_panel(axes[2], image, results["boxes"], results["labels"],
               results["scores"], "Hybrid (Fused)", "#2ecc71")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_single_figure(image, results):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    fig.patch.set_facecolor("white")
    source_colors = {"YOLO": "#e74c3c", "DETR": "#f39c12"}
    ax.imshow(image)
    for box, label, score, source in zip(
        results["boxes"], results["labels"], results["scores"], results["sources"]
    ):
        xmin, ymin, xmax, ymax = box[:4]
        name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
        color = source_colors.get(source, "#2ecc71")
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            edgecolor=color, fill=False, linewidth=2.5,
        ))
        ax.text(
            xmin, ymin - 4, f"{name} {score:.0%} [{source}]",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
        )
    ax.set_title("Hybrid Detection Results", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── Report Generation ──────────────────────────────────────────────────

def generate_report(results):
    n = len(results["boxes"])
    if n == 0:
        return "### No detections found\n\nTry lowering the confidence threshold or uploading a clearer image."

    diseases_found = []
    ripeness_found = []
    for label, score, source in zip(results["labels"], results["scores"], results["sources"]):
        name = DISEASE_NAMES[int(label)] if int(label) < len(DISEASE_NAMES) else f"class_{int(label)}"
        info = DISEASE_INFO.get(name, {})
        entry = {"name": name, "score": score, "source": source, **info}
        if info.get("type") == "Disease":
            diseases_found.append(entry)
        else:
            ripeness_found.append(entry)

    report = f"### Detection Summary\n\n"
    report += f"**{n} object{'s' if n != 1 else ''}** detected "
    report += f"({len(diseases_found)} disease{'s' if len(diseases_found) != 1 else ''}, "
    report += f"{len(ripeness_found)} ripeness state{'s' if len(ripeness_found) != 1 else ''})\n\n"

    if diseases_found:
        report += "---\n\n#### Diseases Detected\n\n"
        for d in sorted(diseases_found, key=lambda x: x["score"], reverse=True):
            report += f"**{d['name']}** — {d['score']:.0%} confidence (from {d['source']})\n"
            report += f"- Severity: {d.get('severity', 'Unknown')}\n"
            report += f"- {d.get('description', '')}\n\n"

    if ripeness_found:
        report += "---\n\n#### Ripeness States\n\n"
        for r in sorted(ripeness_found, key=lambda x: x["score"], reverse=True):
            report += f"**{r['name']}** — {r['score']:.0%} confidence (from {r['source']})\n"
            report += f"- {r.get('description', '')}\n\n"

    return report


# ── Main Predict Function ──────────────────────────────────────────────

def predict(image, confidence_threshold, iou_threshold, show_comparison):
    if image is None:
        return None, "Please upload an image."

    pil_image = Image.fromarray(image).convert("RGB")
    results = run_inference(pil_image, confidence_threshold, iou_threshold)

    if show_comparison:
        fig = create_comparison_figure(pil_image, results)
    else:
        fig = create_single_figure(pil_image, results)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result_image = np.asarray(buf)
    plt.close(fig)

    report = generate_report(results)
    return result_image, report


# ── Gradio Interface ────────────────────────────────────────────────────

TITLE = "Hybrid YOLOv9-DETR Strawberry Disease Detection"

DESCRIPTION = """
Upload a strawberry image to detect **diseases** and **ripeness states** using a hybrid
YOLOv9 + DETR pipeline.

**How it works:** YOLOv9 proposes fast detections, DETR refines them with transformer
attention, and an IoU-based selector fuses the best predictions from each model.

**12 classes** — Angular Leafspot, Anthracnose Fruit Rot, Gray Mold, Leaf Spot,
Powdery Mildew (Fruit & Leaf), and 6 ripeness stages.
"""

ARTICLE = """
<div style="text-align: center; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
    <p>
        <strong>Paper:</strong> Hybrid YOLOv9-DETR for Strawberry Disease Detection (2024)
        &nbsp;|&nbsp;
        <a href="https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease" target="_blank">GitHub Repository</a>
    </p>
    <p style="color: #666; font-size: 0.9em;">
        mAP@0.5: <strong>0.96</strong> (vs. YOLOv9: 0.89, DETR: 0.85)
    </p>
</div>
"""

with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="orange"),
) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Strawberry Image", type="numpy")

            with gr.Accordion("Settings", open=False):
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.5, step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence score for detections",
                )
                iou_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.5, step=0.05,
                    label="IoU Threshold",
                    info="IoU threshold for YOLO-DETR box fusion",
                )
                comparison_toggle = gr.Checkbox(
                    value=True, label="Show 3-panel comparison",
                    info="Show YOLO / DETR / Hybrid side by side",
                )

            detect_btn = gr.Button("Detect Diseases", variant="primary", size="lg")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Detection Results")
            report_output = gr.Markdown(label="Detection Report")

    detect_btn.click(
        fn=predict,
        inputs=[input_image, confidence_slider, iou_slider, comparison_toggle],
        outputs=[output_image, report_output],
    )

    gr.Markdown(ARTICLE)

# ── Launch ──────────────────────────────────────────────────────────────

print("Loading models...")
yolo_model, detr_model, processor = load_models()
print(f"Models loaded on {DEVICE}")

if __name__ == "__main__":
    demo.launch()
