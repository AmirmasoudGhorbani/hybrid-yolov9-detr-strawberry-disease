# Deploying the Gradio Demo to HuggingFace Spaces

Follow these steps to get your live demo running.

## Step 1: Upload Your Weights to HuggingFace Hub

1. Go to https://huggingface.co and sign in (or create an account)
2. Create a new **Model** repository called `hybrid-yolov9-detr-strawberry`
3. Upload your weights with this folder structure:

```
hybrid-yolov9-detr-strawberry/
├── yolo/
│   └── best.pt                          ← your fine-tuned YOLO weights
├── detr_model/
│   ├── config.json                      ← from detr_strawberry_model_extrafinetuned/
│   └── model.safetensors (or pytorch_model.bin)
└── detr_processor/
    └── preprocessor_config.json         ← from detr_strawberry_processor_extrafinetuned/
```

You can upload via the web UI (drag & drop) or the CLI:

```bash
pip install huggingface-hub
huggingface-cli login

# From the folder containing your weights:
huggingface-cli upload AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry yolo/best.pt yolo/best.pt
huggingface-cli upload AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry detr_strawberry_model_extrafinetuned detr_model
huggingface-cli upload AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry detr_strawberry_processor_extrafinetuned detr_processor
```

## Step 2: Create a HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Settings:
   - **Name:** `hybrid-yolov9-detr-strawberry`
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (free) or GPU if available
   - **Visibility:** Public
4. Clone the new Space repo locally:

```bash
git clone https://huggingface.co/spaces/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry
```

5. Copy the contents of this `space/` folder into it:

```bash
cp space/app.py space/requirements.txt space/README.md <your-space-clone>/
```

6. Push:

```bash
cd <your-space-clone>
git add .
git commit -m "Deploy hybrid detection demo"
git push
```

The Space will build automatically. Once done, you'll have a live URL like:
**https://huggingface.co/spaces/AmirmasoudGhorbani/hybrid-yolov9-detr-strawberry**

## Step 3: Update the HF_REPO_ID (if needed)

If your HuggingFace username or repo name differs, edit the `HF_REPO_ID` variable
at the top of `app.py`:

```python
HF_REPO_ID = "YourUsername/your-repo-name"
```

## Troubleshooting

- **Out of memory on CPU Basic:** The DETR model needs ~1.5GB RAM. If the free tier
  isn't enough, upgrade to CPU Upgrade ($0.03/hr) or T4 GPU ($0.10/hr).
- **Weights not found:** Make sure the folder structure in your HF model repo matches
  exactly: `yolo/best.pt`, `detr_model/`, `detr_processor/`.
- **Slow first load:** The first request downloads the weights (~500MB). Subsequent
  requests are cached.
