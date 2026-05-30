import torch
import torch.optim.lr_scheduler as opti
import json
from training.one_epoch_ddp import one_epoch, one_epoch_pred
from training.get_loss_weights import get_loss_weights
import os
from datetime import datetime
import pandas as pd
import torch.distributed as dist

def train_policy(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, training_config, model_config, results_dir_path, scaler):
    
    # Rileviamo se siamo in modalità distribuita e se questo è il processo principale
    is_distributed = dist.is_initialized()
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    if is_main_process:
        print(f"Scaler status -> {'ACTIVE (Mixed Precision)' if scaler is not None else 'INACTIVE (FP32)'}")

    # Il timestamp e la cartella vengono generati e propagati o creati solo dal main process
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
    training_dir_path = os.path.join(results_dir_path, timestamp)
    if is_main_process:
        os.makedirs(training_dir_path, exist_ok=True)

    # In DDP dobbiamo assicurarci che tutte le GPU attendano che il processo 0 abbia creato la cartella
    if is_distributed:
        dist.barrier()

    best_mae_xyz = 100000000 
    best_epoch = 1

    # Salvataggio del file config.json (SOLO processo principale)
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

        # 1. SET_EPOCH PER IL SAMPLER (Fondamentale in DDP)
        # Questo garantisce che ad ogni epoca i dati vengano mescolati in modo diverso tra le GPU
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if is_distributed and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        lambda_actor, lambda_refiner = get_loss_weights(epoch=epoch, total_epoch=num_epochs)
        
        if is_main_process:
            print(f"\n" + "="*30)
            print(f"EPOCH: {epoch+1}/{num_epochs} - LR: {current_lr:.7f} - LAMBDA ACTOR: {lambda_actor:.7f} - LAMBDA REFINER: {lambda_refiner:.7f}")
            print("="*30)

        # One epoch of training 
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

        # One epoch of validation 
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

        # Stampe metriche a schermo (SOLO processo principale)
        if is_main_process:
            print("-" * 80)
            print(f"TRAIN      | Tot Loss: {train_metrics['loss']:.4f} | Actor: {train_metrics['loss_actor']:.4f} | Refiner: {train_metrics['loss_refiner']:.4f}")
            print(f"VALIDATION | Tot Loss: {val_metrics['loss']:.4f} | Actor: {val_metrics['loss_actor']:.4f} | Refiner: {val_metrics['loss_refiner']:.4f}")
            print(f"METRICS  (Validation)  | XYZ Err: {val_metrics['mae_xyz']:.4f} | Gripper Err: {val_metrics['mae_gripper']:.4f} | Cosine Sim Orientation: {val_metrics['cosim_ori']:.4f}")
            print("-" * 80) 

        current_mae_xyz = val_metrics['mae_xyz']

        # 2. SALVATAGGIO DEI CHECKPOINT (SOLO processo principale)
        if is_main_process:
            if current_mae_xyz < best_mae_xyz:
                best_mae_xyz = current_mae_xyz
                best_epoch = epoch+1
                model_save_path = os.path.join(training_dir_path, f"best_model.pth")
                
                # Sotto DDP, salviamo 'model.module.state_dict()' per evitare di includere
                # il prefisso 'module.' all'interno dei pesi salvati.
                state_dict_to_save = model.module.state_dict() if is_distributed else model.state_dict()
                
                checkpoint = {
                    'epoch': best_epoch,
                    'model_state_dict': state_dict_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_config': training_config,
                    'model_config': model_config
                }
                torch.save(checkpoint, model_save_path)
                print(f"🔥 BEST MODEL SAVED! XYZ Err: {current_mae_xyz:.4f} (Epoch {best_epoch})")
            
            # Scrittura della cronologia CSV delle metriche
            df_train = pd.DataFrame(train_history).add_suffix('_train')
            df_val = pd.DataFrame(val_history).add_suffix('_val')

            df_history = pd.concat([df_train, df_val], axis=1)
            df_history.index = range(1, len(df_history) + 1)
            df_history.index.name = 'Epoch'

            csv_save_path = os.path.join(training_dir_path, "metrics.csv")
            df_history.to_csv(csv_save_path)

        # Barriera di sincronizzazione alla fine di ogni epoca (opzionale ma consigliata)
        if is_distributed:
            dist.barrier()

    return training_dir_path


def train_predictor(predictor, vjepa_encoder, train_loader, val_loader, loss_fn, num_epochs, config, results_dir_path, optimizer, device, scaler):

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

    predictor.train()
    # vjepa_encoder è congelato ed è in eval su tutte le GPU
    vjepa_encoder.eval()

    if is_main_process:
        config_dict = { 'hyperparameters': config }
        json_save_path = os.path.join(training_dir_path, "config.json")
        with open(json_save_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    best_vloss = 100000000 
    best_epoch = 1

    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if is_distributed and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        if is_main_process:
            print(f"\n" + "="*30)
            print(f"EPOCH: {epoch+1}/{num_epochs}")
            print("="*30)

        # One epoch of training 
        train_metrics = one_epoch_pred(predictor=predictor, 
                                       vjepa_encoder=vjepa_encoder,
                                       dataloader=train_loader, 
                                       optimizer=optimizer, 
                                       loss_fn=loss_fn, 
                                       device=device,
                                       scaler=scaler
                                  )
    
        # One epoch of validation 
        val_metrics = one_epoch_pred(predictor=predictor, 
                                     vjepa_encoder=vjepa_encoder,
                                     dataloader=val_loader, 
                                     optimizer=optimizer, 
                                     loss_fn=loss_fn, 
                                     device=device,
                                     scaler=scaler,
                                     validation=True
                                )
        
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if is_main_process:
            print("-" * 80)
            print(f"TRAIN      | Tot Loss: {train_metrics['loss']:.4f} | ")
            print(f"VALIDATION | Tot Loss: {val_metrics['loss']:.4f} | ")
            print("-" * 80) 

        current_vloss = val_metrics['loss']

        if is_main_process:
            if current_vloss < best_vloss:
                best_vloss = current_vloss
                best_epoch = epoch+1
                model_save_path = os.path.join(training_dir_path, f"best_model.pth")
                
                state_dict_to_save = predictor.module.state_dict() if is_distributed else predictor.state_dict()
                
                checkpoint = {
                    'epoch': best_epoch,
                    'model_state_dict': state_dict_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, model_save_path)
                print(f"Best model! Saving in: {model_save_path}")

            df_train = pd.DataFrame(train_history).add_suffix('_train')
            df_val = pd.DataFrame(val_history).add_suffix('_val')

            df_history = pd.concat([df_train, df_val], axis=1)
            df_history.index = range(1, len(df_history) + 1)
            df_history.index.name = 'Epoch'

            csv_save_path = os.path.join(training_dir_path, "metrics.csv")
            df_history.to_csv(csv_save_path)
            
        if is_distributed:
            dist.barrier()

    return training_dir_path