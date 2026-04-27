"""
model.py  —  Model definitions for Smart Trash Sorter
  • TrashSorterCNN  : custom CNN built from scratch
  • build_resnet50  : fine-tuned ResNet50 (transfer learning)

The grading interface imports:  TrashSorterCNN as TheModel
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Residual Block ────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """3×3 conv → BN → ReLU → 3×3 conv → BN  +  skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


# ── Custom CNN ────────────────────────────────────────────────────────────────
class TrashSorterCNN(nn.Module):
    """
    Deep CNN with:
      • 4 conv stages  (32 → 64 → 128 → 256 channels)
      • Residual blocks in every stage
      • Spatial Pyramid Pooling (1×1, 2×2, 4×4) for multi-scale features
      • Dropout for regularisation
      • FC classifier head  →  num_classes logits
    Input : (B, 3, 224, 224)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()

        # Stage 1 — 224→112,  3→32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),
        )
        # Stage 2 — 112→56,  32→64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        )
        # Stage 3 — 56→28,  64→128
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128), ResidualBlock(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),
        )
        # Stage 4 — 28→14,  128→256
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResidualBlock(256), ResidualBlock(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Spatial Pyramid Pooling
        self.spp_1 = nn.AdaptiveAvgPool2d(1)   # 256×1   = 256 features
        self.spp_2 = nn.AdaptiveAvgPool2d(2)   # 256×4   = 1 024 features
        self.spp_4 = nn.AdaptiveAvgPool2d(4)   # 256×16  = 4 096 features
        # total flat: 256*(1+4+16) = 5 376

        self.classifier = nn.Sequential(
            nn.Linear(256 * 21, 1024), nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(1024, 256),      nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        p1 = self.spp_1(x).flatten(1)
        p2 = self.spp_2(x).flatten(1)
        p4 = self.spp_4(x).flatten(1)
        return self.classifier(torch.cat([p1, p2, p4], dim=1))


# ── Transfer-Learning Model ───────────────────────────────────────────────────
def build_resnet50(num_classes: int = 6, freeze_backbone: bool = False):
    """ResNet50 pre-trained on ImageNet with a custom FC head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model
