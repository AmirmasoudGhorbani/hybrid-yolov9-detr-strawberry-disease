# Contributing

Thanks for your interest in improving the Hybrid YOLOv9-DETR Strawberry Disease Detector!

## Getting Started

```bash
git clone https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease.git
cd hybrid-yolov9-detr-strawberry-disease
pip install -e .
```

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes** — keep commits focused and descriptive.
3. **Test** your changes with a sample strawberry image before opening a PR.
4. **Open a pull request** against `main` with a clear description.

## Project Structure

- `src/config.py` — all paths and constants live here; edit this first.
- `src/utils.py` — shared datasets, Lightning module, helpers.
- `src/predict.py` — CLI inference entry point.
- `src/retrain_detr.py` — improved DETR retraining script.
- `src/train_*.py`, `src/finetune_*.py` — training pipeline stages.
- `space/` — Gradio app for HuggingFace Spaces.
- `notebooks/` — Colab notebooks for demo and retraining.

## Code Style

- Follow the existing style (no specific formatter enforced).
- Use type hints for public function signatures.
- Keep imports organised: stdlib, third-party, local.

## Reporting Issues

Use the [issue templates](https://github.com/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry-disease/issues/new/choose) for bug reports and feature requests.
