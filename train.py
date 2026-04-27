"""
train.py  —  Training loop for Smart Trash Sorter

Exports:
  train_model  (the_trainer in interface.py)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import config


# ── Single-epoch helpers ──────────────────────────────────────────────────────
def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Main training function (required by interface.py) ────────────────────────
def train_model(model, num_epochs, train_loader, loss_fn, optimizer,
                val_loader=None, model_name="model", save_dir=None):
    """
    Run the full training loop.

    Args:
        model        : nn.Module to train
        num_epochs   : number of epochs
        train_loader : DataLoader for training split
        loss_fn      : loss function (e.g. nn.CrossEntropyLoss)
        optimizer    : torch optimiser
        val_loader   : DataLoader for validation split (optional but recommended)
        model_name   : string used for checkpoint filename and plot title
        save_dir     : folder to save best weights (defaults to config.checkpoint_dir)

    Returns:
        history dict  {train_loss, val_loss, train_acc, val_acc}
    """
    save_dir = save_dir or config.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    device = config.device
    model  = model.to(device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}  |  Device: {device}")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, loss_fn, optimizer, device)

        if val_loader is not None:
            vl_loss, vl_acc = _evaluate(model, val_loader, loss_fn, device)
        else:
            vl_loss, vl_acc = float("nan"), float("nan")

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt_path = os.path.join(save_dir, f"best_{model_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            marker = "  ← saved"
        else:
            marker = ""

        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"| Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  "
              f"| Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}{marker}")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    _plot_history(history, model_name)
    return history


# ── Plot helper ───────────────────────────────────────────────────────────────
def _plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(f"history_{model_name}.png", dpi=150)
    plt.show()
    print(f"Saved training plot → history_{model_name}.png")
