# hybrid-yolov9-detr-strawberry-disease
# 🍓 Hybrid YOLOv9-DETR for Strawberry Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv9](https://img.shields.io/badge/YOLOv9-Object%20Detection-red)](https://github.com/WongKinYiu/yolov9)
[![DETR](https://img.shields.io/badge/DETR-Transformer%20Detection-green)](https://github.com/facebookresearch/detectron2)

## 📌 Overview
This repository contains the **Hybrid YOLOv9-DETR Model** for **strawberry disease detection**, integrating the strengths of **YOLOv9 (fast object detection)** and **DETR (accurate bounding box refinement)**. This hybrid approach significantly improves disease detection accuracy while maintaining real-time inference capabilities.

🚀 **Key Features:**
- **Combines YOLOv9 and DETR** for enhanced object detection.
- **Non-end-to-end hybrid approach** for better computational efficiency.
- **Strawberry disease dataset** from **Roboflow** (12 classes).
- **Achieves mAP@0.5 = 0.96**, outperforming standalone YOLO and DETR.
- **Future-ready for edge deployment in precision agriculture.**

## 📂 Dataset
The dataset used in this study contains **12 classes of strawberry diseases and ripeness states**, structured in the YOLO format:
dataset/ │── train/ │ ├── images/ │ ├── labels/ │── val/ │ ├── images/ │ ├── labels/ │── test/ │ ├── images/ │ ├── labels/ │── dataset.yaml
📌 **Dataset Link:** [Roboflow - Strawberry Disease Dataset](https://universe.roboflow.com/esther-strawberry/strawberry-fruit-and-leaf-ripeness-and-disease)

---

## 🛠 Installation
Clone this repository:
git clone https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease.git
cd hybrid-yolov9-detr-strawberry-disease

Install dependencies:
pip install -r requirements.txt
pip install torch torchvision

🚀 Model Training & Evaluation

1️⃣ Train the Hybrid Model
To train the YOLOv9-DETR hybrid model, use:

python train.py --config yolov9-detr-config.yaml

2️⃣ Run Inference
To test the model on a single image:

python inference.py --image data/test.jpg --weights yolov9-detr-best.pth

3️⃣ Evaluate Model Performance

python evaluate.py --weights yolov9-detr-best.pth
📊 Results & Performance
Metric	YOLOv9	DETR	Hybrid YOLOv9-DETR
mAP@0.5	0.89	0.85	0.96
mAP@0.75	0.79	0.81	0.92
Precision	0.68	0.64	0.82
Recall	0.73	0.69	0.85
📌 Key Findings:

The hybrid model outperforms individual models in all key metrics.
DETR refines YOLO’s bounding boxes, leading to improved accuracy.
Hybrid models can be optimized for real-time deployment in agriculture.

Future Work
Optimizing for real-time edge deployment on NVIDIA Jetson & Raspberry Pi.
Fine-tuning hybrid feature fusion techniques for better class differentiation.
Expanding the dataset with multi-spectral imaging for improved disease detection.

🤝 Contributing
Feel free to fork this repository, submit issues, and contribute new features or optimizations!

📜 Citation
If you use this project for research, please cite:

bibtex

@misc{ghorbani2024,
  author = {Ghorbani, Amir},
  title = {Hybrid YOLOv9-DETR for Strawberry Disease Detection},
  year = {2024},
  url = {https://github.com/your-username/hybrid-yolov9-detr-strawberry-disease},
  note = {Accessed: February 11, 2025}
}
