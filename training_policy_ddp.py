import os
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
# --- MODIFICATO: Aggiunto l'import di MixedPrecision per FSDP ---
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
# ----------------------------------------------------------------------
from torch.utils.data.distributed import DistributedSampler
from Dataset.PolicyDataset2 import PolicyDataset
from model.TransformerActionJEPA2 import TransformerActionJEPA
from training.train_ddp import train_policy
import numpy as np
from torch.amp import GradScaler

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':

    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0) 

    seed = 46 + rank 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    device = f"cuda:{local_rank}"

    with open('./config/config_training_policy.json', 'r') as f:
        training_config = json.load(f)

    with open('./config/model_config.json', 'r') as f:
        model_config = json.load(f)

    NUM_EPOCHS = training_config['num_epochs']
    BATCH_SIZE = training_config['batch_size']
    LEARNING_RATE = training_config['learning_rate']
    DATASETS = training_config['datasets']   
    TASK_IDS = training_config['task_ids'] 
    MIXED_PRECISION = training_config['mixed_precision'] 
    NUM_WORKERS = training_config['num_workers']
    PREFETCH_FACTOR = training_config['prefetch_factor']

    FINETUNED_PRED = model_config['finetuned_pred']
    MAX_LENGTH = model_config['max_length'] 
    POLICY = model_config['policy'] 
    NUM_FRAMES = model_config['num_frames']
    EMBED_DIM = model_config['embed_dim']
    FROZEN_BACKBONE = model_config['frozen_backbone']
    TRANSFORMER_LAYERS = model_config['transformer_layers']
    TRANSFORMER_HEADS = model_config['transformer_heads']
    TRANSFORMER_FF_DIM = model_config['transformer_ff_dim']
    TRANSFORMER_DROPOUT = model_config['transformer_dropout']
    MLP_HIDDEN_DIMS = model_config['mlp_hidden_dims']
    MLP_DROPOUT = model_config['mlp_dropout']

    if is_main_process:
        print("="*40)
        print(f"✅ Distributed Training Config Created! World Size (GPUs): {world_size}")
        print(f"Epochs: {NUM_EPOCHS} | Batch Size per GPU: {BATCH_SIZE} (Total: {BATCH_SIZE * world_size})")
        print(f"LR: {LEARNING_RATE} ")
        print(f"Dataset: {DATASETS}")
        print(f"Tasks: {TASK_IDS}")
        print(f"Policy type: {POLICY}")
        print(f"Mixed Precision: {MIXED_PRECISION}")
        print("="*40)

    dataset = PolicyDataset(datasets=DATASETS,
                            task_ids=TASK_IDS,
                            num_frames=NUM_FRAMES,
                            full_load_ram=True
                            )
    
    if is_main_process:
        print("Finish Loading Dataset...")

    train_percentage = 0.7
    val_percentage = 0.2

    train_size = int(train_percentage*len(dataset))
    val_size = int(val_percentage*len(dataset))
    test_size = len(dataset) - train_size - val_size

    g = torch.Generator()
    g.manual_seed(46)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset, 
        lengths=[train_size, val_size, test_size],
        generator=g
    )

    if is_main_process:
        print(f"Total dataset size (windows): {len(dataset)}, Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}")

    checkpoints_path = "./checkpoints"
    vjepa_path = os.path.join(checkpoints_path, "facebook/vjepa2-vitg-fpc64-256")
    if FINETUNED_PRED:
        predictor_path = 'results/predictor/2026_05_15__14_49/best_model.pth'
    else:
        predictor_path = os.path.join(checkpoints_path, "facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar")
    clip_path = os.path.join(checkpoints_path, "openai/clip-vit-large-patch14")

    model = TransformerActionJEPA(
        vjepa_encoder_path=vjepa_path,
        vjepa_predictor_path=predictor_path,
        clip_model_path=clip_path,
        num_frames=NUM_FRAMES,
        embed_dim=EMBED_DIM,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_ff_dim=TRANSFORMER_FF_DIM,
        transformer_dropout=TRANSFORMER_DROPOUT,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
        mlp_dropout=MLP_DROPOUT,
        frozen_backbone=FROZEN_BACKBONE,
        finetuned_pred=FINETUNED_PRED,
        device='cpu',
    )
    model.to('cpu')

    if hasattr(model.language_backbone.clip_model, "logit_scale"):
        # Unsqueeze(0) aggiunge la dimensione mancante richiesta da FSDP
        model.language_backbone.clip_model.logit_scale = nn.Parameter(
            model.language_backbone.clip_model.logit_scale.unsqueeze(0)
        )
    # ==========================================================================

    # 2. Ora lo passi a FSDP, che non troverà più parametri a 0 dimensioni e partirà felice
    auto_wrap_policy = size_based_auto_wrap_policy

    # --- MODIFICATO: Configurazione della precisione mista fissa per i pesi FSDP ---
    fsdp_mixed_precision = MixedPrecision(
        param_dtype=torch.float16,     # Riduce l'occupazione dei pesi in VRAM a 16-bit
        reduce_dtype=torch.float16,    # Esegue la riduzione dei gradienti a 16-bit
        buffer_dtype=torch.float16,
    )

    auto_wrap_policy = size_based_auto_wrap_policy
    
    model = FSDP(
        model, 
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, # Sfrutta ZeRO-2 per gradienti e ottimizzatore
        cpu_offload=None, # <--- IMPOSTA A NONE (Rimuove il blocco di controllo CPU/CUDA)
        mixed_precision=fsdp_mixed_precision,
        sync_module_states=True
    )

    if is_main_process:
        model._fsdp_wrapped_module.print_model_info() 

    results_dir_path = "./results/policy"
    if is_main_process:
        os.makedirs(results_dir_path, exist_ok=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if MIXED_PRECISION:
        scaler = GradScaler()
    else:
        scaler = None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
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
        model_config=model_config,
        device=device,
        scaler=scaler,
        results_dir_path=results_dir_path,
    )

    cleanup_ddp()