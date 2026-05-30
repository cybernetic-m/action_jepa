import os
import random
import json
import torch
import torch.optim.lr_scheduler as opti
from training.one_epoch_ddp import one_epoch, one_epoch_pred
from training.get_loss_weights import get_loss_weights
import os
from datetime import datetime
import pandas as pd
import torch.distributed as dist
# --- MODIFICATO: Rimossa la chiamata a DDP e aggiunti gli import FSDP per il salvataggio ---
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
# -----------------------------------------------------------------------------------------

def train_policy(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, training_config, model_config, results_dir_path, scaler):
    
    is_distributed = dist.is_initialized()
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    if is_main_process:
        print(f"Scaler status -> {'ACTIVE (Mixed Precision)' if scaler is not None else 'INACTIVE (FP32)'}")

    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
    training_dir_path = os.path.join(results_dir_path, timestamp)
    if is_main_process:
        os.makedirs(training_dir_path, exist_ok=True)

    if is_distributed:
        dist.barrier()

    best_mae_xyz = 100000000 
    best_epoch = 1

    if is_main_process:
        config_dict = {
            'training': training_config,
            'model': model_config
        }
        json_save_path = os.path.join(training_dir_path, "config.json")
        with open(json_save_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    train_history = []
    val_history = []

    scheduler = opti.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):

        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if is_distributed and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        lambda_actor, lambda_refiner = get_loss_weights(epoch=epoch, total_epoch=num_epochs)
        
        if is_main_process:
            print(f"\n" + "="*30)
            print(f"EPOCH: {epoch+1}/{num_epochs} - LR: {current_lr:.7f} - LAMBDA ACTOR: {lambda_actor:.7f} - LAMBDA REFINER: {lambda_refiner:.7f}")
            print("="*30)

        train_metrics = one_epoch(model=model, 
                                  dataloader=train_loader, 
                                  optimizer=optimizer, 
                                  loss_fn=loss_fn, 
                                  device=device,
                                  scaler=scaler,
                                  lambda_actor=lambda_actor, 
                                  lambda_refiner=lambda_refiner
                                  )
        train_metrics['lr'] = current_lr    
        train_metrics['lambda_actor'] = lambda_actor
        train_metrics['lambda_refiner'] = lambda_refiner

        val_metrics = one_epoch(model=model, 
                                dataloader=val_loader, 
                                optimizer=optimizer, 
                                loss_fn=loss_fn, 
                                device=device,
                                scaler=scaler,
                                lambda_actor=lambda_actor, 
                                lambda_refiner=lambda_refiner,
                                validation=True
                                )
        
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process:
            print("-" * 80)
            print(f"TRAIN      | Tot Loss: {train_metrics['loss']:.4f} | Actor: {train_metrics['loss_actor']:.4f} | Refiner: {train_metrics['loss_refiner']:.4f}")
            print(f"VALIDATION | Tot Loss: {val_metrics['loss']:.4f} | Actor: {val_metrics['loss_actor']:.4f} | Refiner: {val_metrics['loss_refiner']:.4f}")
            print(f"METRICS  (Validation)  | XYZ Err: {val_metrics['mae_xyz']:.4f} | Gripper Err: {val_metrics['mae_gripper']:.4f} | Cosine Sim Orientation: {val_metrics['cosim_ori']:.4f}")
            print("-" * 80) 

        current_mae_xyz = val_metrics['mae_xyz']

        # --- MODIFICATO: SALVATAGGIO DEI CHECKPOINT BLINDATO PER FSDP ---
        if current_mae_xyz < best_mae_xyz:
            best_mae_xyz = current_mae_xyz
            best_epoch = epoch + 1
            model_save_path = os.path.join(training_dir_path, f"best_model.pth")
            
            if is_distributed:
                # Diciamo a FSDP di riunire i pesi sulla CPU solo per il salvataggio
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict_to_save = model.state_dict()
            else:
                state_dict_to_save = model.state_dict()

            if is_main_process:
                checkpoint = {
                    'epoch': best_epoch,
                    'model_state_dict': state_dict_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_config': training_config,
                    'model_config': model_config
                }