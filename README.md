🍓 Hybrid YOLOv9–DETR for Strawberry Disease Detection

Python 
YOLOv9 
DETR 
License: MIT

A hybrid object-detection pipeline that pairs YOLOv9 (fast detection) with DETR 
(transformer-based refinement) to detect strawberry diseases and ripeness states. YOLO 
proposes boxes; DETR refines them; an IoU-based selector keeps the best of each. The 
combination beats either model alone — mAP@0.5 = 0.96.

Thesis project. The code is organised as a sequence of training/evaluation scripts that 
were developed and run in Google Colab (paths point at Google Drive — see 
Configuration).



📊 Results

MetricYOLOv9DETRHybridmAP@0.50.890.850.96mAP@0.750.790.810.92Precision0.680.640.82Recall0.730.690.85
Key findings
– The hybrid model outperforms both standalone models on every metric.
– DETR refines YOLO's bounding boxes, improving localisation.
– The non-end-to-end design keeps inference efficient enough for real-time use.


📂 Dataset

12 classes of strawberry diseases and ripeness states, in COCO format for DETR and YOLO 
format for YOLOv9:
dataset/
├── train/   images/  labels (or _annotations.coco.json)
├── val/     images/  labels
└── test/    images/  labels

Classes: Angular Leafspot · Anthracnose Fruit Rot · Early-Turning · Gray Mold · 
Green-Strawberry · Late-Turning · Leaf Spot · Powdery Mildew Fruit · Powdery Mildew Leaf · 
Red-Turning · Turning · White-Strawberry

📌 Source: Roboflow — Strawberry Disease Dataset


🗂 Repository layout

hybrid-yolov9-detr-strawberry-disease/
├── README.md
├── LICENSE
├── requirements.txt
└── src/
    ├── train_yolo.py            # 1. train YOLOv9s from scratch
    ├── finetune_yolo.py         # 2. fine-tune the best YOLO weights (lower LR)
    ├── train_detr.py            # 3. train DETR (facebook/detr-resnet-50)
    ├── optimize_detr.py         # 4. fine-tune DETR + StepLR scheduler + confusion matrix
    ├── finetune_detr_extra.py   # 5. extra DETR fine-tuning pass
    ├── evaluate_detr.py         # 6. evaluate DETR on the test set (mAP, P/R/F1)
    └── hybrid_inference.py      # 7. YOLO + DETR fusion via IoU-based box selection


▶️ Pipeline & run order

The scripts form a pipeline — each stage consumes the weights produced by the previous one.
#ScriptInputProduces1train_yolo.pydataset (YOLO format)YOLOv9 best.pt2finetune_yolo.pyYOLO best.ptfine-tuned YOLO weights3train_detr.pydataset (COCO) + facebook/detr-resnet-50detr_strawberry_model_24optimize_detr.pydetr_strawberry_model_2..._model_finetuned5finetune_detr_extra.py..._model_finetuned..._model_extrafinetuned6evaluate_detr.py..._model_extrafinetunedmetrics (mAP, P/R/F1, confusion matrix)7hybrid_inference.pyfine-tuned YOLO + ..._model_extrafinetunedfused predictions + visualisationspython src/train_yolo.py
python src/finetune_yolo.py
python src/train_detr.py
python src/optimize_detr.py
python src/finetune_detr_extra.py
python src/evaluate_detr.py
python src/hybrid_inference.py


🛠 Installation

git clone https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease.git
cd hybrid-yolov9-detr-strawberry-disease
pip install -r requirements.txt

A CUDA-capable GPU is strongly recommended for training.


⚙️ Configuration

These scripts were authored in Google Colab, so paths default to Google Drive, e.g.:
train_path = '/content/drive/MyDrive/dataset/dataset json format/train'
yolo_weights = '/content/drive/MyDrive/yolov9_lastfinetune_results/.../best.pt'

Before running locally, edit the path constants near the top of each script to point at 
your dataset and weights directories.


🔭 Future work

– Real-time edge deployment on NVIDIA Jetson / Raspberry Pi
– Better feature-fusion between the two detectors for finer class separation
– Multi-spectral imaging to improve early disease detection

📜 Citation

@misc{ghorbani2024hybrid,
  author = {Ghorbani, Amir},
  title  = {Hybrid YOLOv9-DETR for Strawberry Disease Detection},
  year   = {2024},
  url    = {https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease}
}

📄 License

Released under the MIT License.
