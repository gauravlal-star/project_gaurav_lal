# interface.py
# ── DO NOT rename the aliases on the right-hand side ──────────────────────────
# The grading script imports exactly these names.

# replace TrashSorterCNN with the name of your model
from model import TrashSorterCNN as TheModel

# the function inside train.py that runs the training loop
from train import train_model as the_trainer

# the function inside predict.py that runs inference on a list of image paths
from predict import classify_waste as the_predictor

# your custom Dataset class
from dataset import WasteImageDataset as TheDataset

# your DataLoader factory function
from dataset import get_dataloader as the_dataloader

# hyperparameters from config.py
from config import batch_size as the_batch_size
from config import epochs as total_epochs
