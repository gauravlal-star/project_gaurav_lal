"""
predict.py  —  Inference for Smart Trash Sorter

Exports:
  classify_waste  (the_predictor in interface.py)
"""

import os
from typing import List

import torch
from PIL import Image
from torchvision import transforms

import config

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

if not list_of_img_paths:
        return []
# ── Transform for inference (no augmentation) ─────────────────────────────────
_infer_tf = transforms.Compose([
    transforms.Resize((config.resize_y, config.resize_x)),
    transforms.ToTensor(),
    transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
])


# ── Core prediction function (required by interface.py) ───────────────────────
def classify_waste(list_of_img_paths: List[str], model=None,
                   weights_path: str = None) -> List[str]:
    """
    Run inference on a batch of image file paths.

    Args:
        list_of_img_paths : list of paths to .jpg/.png images
        model             : nn.Module (optional — loaded from weights if None)
        weights_path      : path to .pth checkpoint
                            (defaults to checkpoints/best_resnet50.pth)

    Returns:
        List of predicted class label strings, one per input image.
    """
    device = config.device

    # ── Load model if not passed in ──────────────────────────────────────────
    if model is None:
        from model import build_resnet50
        weights_path = weights_path or os.path.join(
            config.checkpoint_dir, "best_resnet50.pth")
        model = build_resnet50(num_classes=config.num_classes)
        model.load_state_dict(
            torch.load(weights_path, map_location=device))
        print(f"[predict] Loaded weights from {weights_path}")

    model = model.to(device)
    model.eval()

    # ── Build batch tensor ───────────────────────────────────────────────────
    tensors = []
    for path in list_of_img_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(_infer_tf(img))
    batch = torch.stack(tensors).to(device)   # (N, 3, H, W)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(batch)                  # (N, num_classes)
        probs  = torch.softmax(logits, dim=1)  # (N, num_classes)
        preds  = probs.argmax(dim=1)           # (N,)

    labels = [CLASSES[p.item()] for p in preds]
    return labels


# ── Pretty single-image prediction (for demo / testing) ───────────────────────
def predict_single(model, image_path: str):
    """Print a confidence bar for one image."""
    result = classify_waste([image_path], model=model)
    label  = result[0]

    # Recompute probs for the bar chart
    img    = Image.open(image_path).convert("RGB")
    tensor = _infer_tf(img).unsqueeze(0).to(config.device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    print(f"\nImage     : {os.path.basename(image_path)}")
    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {probs[CLASSES.index(label)]*100:.2f}%")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 30)
        print(f"  {cls:<12} {probs[i]*100:5.1f}%  {bar}")
    return label
