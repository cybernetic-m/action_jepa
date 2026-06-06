import os
import random
import json
from Dataset.PolicyDataset2 import PolicyDataset
from model.TransformerActionJEPA2 import TransformerActionJEPA
from training.train import train_policy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.amp import GradScaler

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


    # Loading the config.json files
    with open('./config/config_training_policy.json', 'r') as f:
        training_config = json.load(f)

    with open('./config/model_config.json', 'r') as f:
        model_config = json.load(f)

    # Training Hyperparameters 
    NUM_EPOCHS = training_config['num_epochs']
    BATCH_SIZE = training_config['batch_size']
    LEARNING_RATE = training_config['learning_rate']
    DATASETS = training_config['datasets']   # it can be  a list of "libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"...
    TASK_IDS = training_config['task_ids'] # it can be a list of task_ids (from 0 to 9) depending on how much tasks of the datasets you want to train your model
    MIXED_PRECISION = training_config['mixed_precision'] # it can be true or false depending if you want to use float16 (instead of float32) using tensor cores of your gpu
    NUM_WORKERS = training_config['num_workers']
    PREFETCH_FACTOR = training_config['prefetch_factor']

    # Model Hyperparameters
    FINETUNED_PRED = model_config['finetuned_pred']
    MAX_LENGTH = model_config['max_length'] # it is the resulting tokens length after CLIP text encoder
    POLICY = model_config['policy'] # it can be "mlp" or "transformer"
    NUM_FRAMES = model_config['num_frames']
    ACTION_CHUNK_SIZE = model_config['action_chunk_size']
    EMBED_DIM = model_config['embed_dim']
    FROZEN_BACKBONE = model_config['frozen_backbone']
    TRANSFORMER_LAYERS = model_config['transformer_layers']
    TRANSFORMER_HEADS = model_config['transformer_heads']
    TRANSFORMER_FF_DIM = model_config['transformer_ff_dim']
    TRANSFORMER_DROPOUT = model_config['transformer_dropout']
    MLP_HIDDEN_DIMS = model_config['mlp_hidden_dims']
    MLP_DROPOUT = model_config['mlp_dropout']

    print("="*40)
    print(f"✅ Training config created!")
    print(f"Num Frames: {NUM_FRAMES}")
    print(f"Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"LR: {LEARNING_RATE} ")
    print(f"Dataset: {DATASETS}")
    print(f"Tasks: {TASK_IDS}")
    print(f"Fine Tuned predictor: {FINETUNED_PRED}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Policy type: {POLICY}")
    print(f"Mixed Precision: {MIXED_PRECISION}")
    print("="*40)

    dataset = PolicyDataset(datasets=DATASETS,
                        task_ids = TASK_IDS,
                        num_frames=NUM_FRAMES,
                        action_chunk_size=ACTION_CHUNK_SIZE,
                        full_load_ram=True
                        )
    print("Finish Loading Dataset...")

    train_percentage = 0.7
    val_percentage = 0.2

    train_size = int(train_percentage*len(dataset))
    val_size = int(val_percentage*len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])

    print(f"Total dataset size (windows): {len(dataset)}, Train Size (windows): {len(train_dataset)}, Validation Size (windows): {len(val_dataset)}, Test Size (windows): {len(test_dataset)}")

    checkpoints_path = "./checkpoints"
    # Path for all the models
    vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
    if FINETUNED_PRED:
        predictor_path = 'results/predictor/2026_05_15__14_49/best_model.pth'
    else:
        predictor_path = os.path.join(checkpoints_path,"facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar")
    clip_path = os.path.join(checkpoints_path,"openai/clip-vit-large-patch14")

    model = TransformerActionJEPA(
        vjepa_encoder_path=vjepa_path,
        vjepa_predictor_path=predictor_path,
        clip_model_path=clip_path,
        num_frames=NUM_FRAMES,
        action_chunk_size=ACTION_CHUNK_SIZE,   
        embed_dim = EMBED_DIM,
        transformer_layers = TRANSFORMER_LAYERS,
        transformer_heads = TRANSFORMER_HEADS,
        transformer_ff_dim = TRANSFORMER_FF_DIM,
        transformer_dropout = TRANSFORMER_DROPOUT,
        mlp_hidden_dims = MLP_HIDDEN_DIMS,
        mlp_dropout = MLP_DROPOUT,
        frozen_backbone = FROZEN_BACKBONE,
        finetuned_pred = FINETUNED_PRED,
        device=device,
    ).to(device)

    model.print_model_info()

    # Name of the directory for the results
    results_dir_path = "./results/policy"
    os.makedirs(results_dir_path, exist_ok=True)

    # Definition of the Categorical Cross Entropy Loss
    loss_fn = nn.MSELoss()
    # Definition of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    if MIXED_PRECISION:
        scaler = GradScaler()
    else:
        scaler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    
    training_dir_path = train_policy(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        training_config=training_config,
        model_config = model_config,
        device=device,
        scaler=scaler,
        results_dir_path=results_dir_path,
    )
    




