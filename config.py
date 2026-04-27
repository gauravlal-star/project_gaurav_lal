import torch

# ── Hyperparameters ──────────────────────────────────────────────────────────
epochs     = 40
batch_size = 32
lr         = 3e-4
weight_decay = 1e-4
train_split  = 0.7
val_split    = 0.15
# test_split = remainder (0.15)

# ── Image dimensions ─────────────────────────────────────────────────────────
image_size    = 224          # model expects 224×224 RGB
resize_x      = image_size
resize_y      = image_size
input_channels = 3

# ── Paths ────────────────────────────────────────────────────────────────────
# Point this to the inner folder that contains the 6 class sub-folders:
#   cardboard/ glass/ metal/ paper/ plastic/ trash/
data_dir   = "./data"
checkpoint_dir = "./checkpoints"

# ── Misc ─────────────────────────────────────────────────────────────────────
num_classes = 6
device      = "cuda" if torch.cuda.is_available() else "cpu"
seed        = 42

# ImageNet normalisation stats (used by both models)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
