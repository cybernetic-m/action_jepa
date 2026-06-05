import os
import random
import json
from Dataset.PredictorDataset import PredictorDataset
from model.modules.PredictorAC import PredictorAC
from model.modules.VJEPAEncoder import VJEPAEncoder
from old.train import train_predictor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.amp import GradScaler
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # REPRODUCIBILITY
    seed = 46

    # Set seed for torch, numpy and random libraries
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # Set the devide mode on GPU (if available CUDA for Nvidia and  MPS for Apple Silicon) or CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


    # Loading the config.json file
    with open('config_predictor.json', 'r') as f:
        config = json.load(f)

    # Hyperparameters definition
    NUM_EPOCHS = config['num_epochs']
    BATCH_SIZE = config['batch_size']
    NUM_FRAMES = config['num_frames']
    LEARNING_RATE = config['learning_rate']
    DATASETS = config['datasets']   # it can be  a list of "libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"...
    TASK_IDS = config['task_ids'] # it can be a list of task_ids (from 0 to 9) depending on how much tasks of the datasets you want to train your model
    MIXED_PRECISION = config['mixed_precision'] # it can be true or false depending if you want to use float16 (instead of float32) using tensor cores of your gpu
    NUM_WORKERS = config['num_workers']

    print("="*40)
    print(f"✅ Training config created!")
    print(f"Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"LR: {LEARNING_RATE}")
    print(f"Dataset: {DATASETS}")
    print(f"Dataset: {TASK_IDS}")
    print(f"Mixed Precision: {MIXED_PRECISION}")
    print("="*40)

checkpoints_path = "./checkpoints"
# Path for all the models
vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
vjepa_pred_path = os.path.join(checkpoints_path,"facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar")

predictor = PredictorAC(
    model_path=vjepa_pred_path,
    frozen=False,
    num_frames = NUM_FRAMES,
    device=device
).to(device)

vjepa_encoder = VJEPAEncoder(
    model_path=vjepa_path,
    device=device
).to(device)

data_dir ='./resampled_data'

dataset = PredictorDataset(data_dir=data_dir,
                        selected_tasks=DATASETS,
                        task_ids = TASK_IDS,
                        preprocess_fn = vjepa_encoder.preprocess_frames
                        )

train_percentage = 0.7
val_percentage = 0.2

train_size = int(train_percentage*len(dataset))
val_size = int(val_percentage*len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])

print(f"Total dataset size (windows): {len(dataset)}, Train Size (windows): {len(train_dataset)}, Validation Size (windows): {len(val_dataset)}, Test Size (windows): {len(test_dataset)}")

with open('config_predictor.json', 'r') as f:
    config = json.load(f)

# Name of the directory for the results
results_dir_path = "./results/predictor"
os.makedirs(results_dir_path, exist_ok=True)

# Definition of the Categorical Cross Entropy Loss
loss_fn = torch.nn.L1Loss()
# Definition of the optimizer
optimizer = torch.optim.AdamW(predictor.parameters(), lr = LEARNING_RATE)

if MIXED_PRECISION:
    scaler = GradScaler()
else:
    scaler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

training_dir_path = train_predictor(
    predictor=predictor,
    vjepa_encoder=vjepa_encoder,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=NUM_EPOCHS,
    config=config,
    device=device,
    scaler=scaler,
    results_dir_path=results_dir_path,
)

