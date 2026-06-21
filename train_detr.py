"""
train_detr.py — Stage 3 of the pipeline.

Trains DETR (facebook/detr-resnet-50) on the strawberry dataset in COCO format using
PyTorch Lightning, then saves it as `detr_strawberry_model_2`. Includes a quick
inference + ground-truth visualisation at the end as a sanity check.

Paths below default to Google Colab / Google Drive — edit them for your environment.
"""

# --- Colab setup (uncomment when running in a fresh Colab runtime) -----------
# !pip install torch torchvision
# !pip install cython pycocotools
# !pip install -U numpy
# !pip install transformers pytorch-lightning pycocotools
# !pip install numpy==1.23.5
# !pip install datasets
# !pip install -q timm
# !pip install -i https://test.pypi.org/simple/ supervision==0.3.0
# !pip install -q roboflow

import os
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

# Define paths to your dataset
train_path = '/content/drive/MyDrive/dataset/dataset json format/train'
val_path = '/content/drive/MyDrive/dataset/dataset json format/valid'
annotation_file = '_annotations.coco.json'

# Setup DetrImageProcessor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


# Custom CocoDetection class for your dataset
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, annotation_file)
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


# Create training and validation datasets
train_dataset = CocoDetection(img_folder=train_path, processor=processor)
val_dataset = CocoDetection(img_folder=val_path, processor=processor, train=False)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")


# DataLoader with custom collate function
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True, num_workers=11)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4, num_workers=11)


# Define the PyTorch Lightning model
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=13,  # 12 classes + 1 for 'no object'
            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        for label in labels:
            label["class_labels"] = torch.clamp(label["class_labels"], min=0, max=self.model.config.num_labels - 2)
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)


def plot_results(pil_img, scores, labels, boxes, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]] * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{model.model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Train the model using PyTorch Lightning
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_float32_matmul_precision('medium')
    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions for debugging
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    trainer = pl.Trainer(max_epochs=30, devices=1, accelerator='gpu', log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the trained model
    model.model.save_pretrained("/content/drive/MyDrive/detr_strawberry_model_2")
    processor.save_pretrained("/content/drive/MyDrive/detr_strawberry_processor_2")

    # --- Sanity-check inference on a validation image ---
    image_id = val_dataset.coco.getImgIds()[0]
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(val_path, image['file_name']))

    pixel_values, _ = val_dataset[0]
    pixel_values = pixel_values.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model.model(pixel_values=pixel_values)

    width, height = image.size
    results = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.9)[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'], model)

    # --- Visualise a random ground-truth annotation to confirm the dataset loaded ---
    image_ids = train_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print(f"Image #{image_id}")

    image = train_dataset.coco.loadImgs(image_id)[0]
    annotations = train_dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(train_dataset.root, image['file_name'])
    image = cv2.imread(image_path)

    categories = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}

    plt.figure(figsize=(10, 10))
    for ann in annotations:
        x, y, w, h = ann['bbox']
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        cv2.putText(image, id2label[ann['category_id']], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
