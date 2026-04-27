"""
main.py  —  End-to-end runner for Smart Trash Sorter

Usage:
    python main.py                        # train both models
    python main.py --model resnet50       # train only ResNet50
    python main.py --predict path/img.jpg # run inference on one image
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model import TrashSorterCNN, build_resnet50
from dataset import load_all_splits
from train import train_model
from predict import classify_waste, predict_single


@torch.no_grad()
def evaluate_on_test(model, test_loader, model_name):
    device = config.device
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    actual_classes = test_loader.dataset.dataset.classes
    print(f"\n{'='*55}\n  Test Report — {model_name}\n{'='*55}")
    print(classification_report(all_labels, all_preds,
                                target_names=actual_classes,
                                labels=list(range(len(actual_classes)))))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=actual_classes, yticklabels=actual_classes)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_{model_name}.png", dpi=150)
    # plt.show() 
    print(f"Saved confusion matrix → confusion_{model_name}.png")


def run_training(model_choice="both"):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_all_splits()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Model A: Custom CNN ───────────────────────────────────────────────────
    if model_choice in ("both", "custom"):
        custom_model = TrashSorterCNN(num_classes=config.num_classes)
        print(f"Custom CNN parameters: "
              f"{sum(p.numel() for p in custom_model.parameters()):,}")
        optimizer_a = optim.AdamW(custom_model.parameters(),
                                  lr=config.lr,
                                  weight_decay=config.weight_decay)
        train_model(custom_model, config.epochs,
                    train_loader, loss_fn, optimizer_a,
                    val_loader=val_loader, model_name="custom_cnn")
        ckpt = os.path.join(config.checkpoint_dir, "best_custom_cnn.pth")
        custom_model.load_state_dict(
            torch.load(ckpt, map_location=config.device))
        evaluate_on_test(custom_model, test_loader, "custom_cnn")

    # ── Model B: ResNet50 ─────────────────────────────────────────────────────
    if model_choice in ("both", "resnet50"):
        resnet_model = build_resnet50(num_classes=config.num_classes)
        optimizer_b  = optim.AdamW(resnet_model.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)
        train_model(resnet_model, config.epochs,
                    train_loader, loss_fn, optimizer_b,
                    val_loader=val_loader, model_name="resnet50")
        ckpt = os.path.join(config.checkpoint_dir, "best_resnet50.pth")
        resnet_model.load_state_dict(
            torch.load(ckpt, map_location=config.device))
        evaluate_on_test(resnet_model, test_loader, "resnet50")

        # Goal 4 — test on unseen images in data/
        print(f"\n{'='*55}\n  Goal 4: Inference on data/ images\n{'='*55}")
        
        data_imgs = [
            os.path.join(config.data_dir, f)
            for f in os.listdir(config.data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
                    
        data_imgs = []
        for root, dirs, files in os.walk(config.data_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    data_imgs.append(os.path.join(root, f))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="both",
                        choices=["both", "custom", "resnet50"])
    parser.add_argument("--predict", default=None,
                        help="Path to a single image for inference")
    args = parser.parse_args()

    if args.predict:
        # Quick inference without training
        from model import build_resnet50
        m = build_resnet50(num_classes=config.num_classes)
        ckpt = os.path.join(config.checkpoint_dir, "best_resnet50.pth")
        m.load_state_dict(torch.load(ckpt, map_location=config.device))
        predict_single(m, args.predict)
    else:
        run_training(args.model)
