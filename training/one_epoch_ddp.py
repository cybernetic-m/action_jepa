from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist

def reduce_tensor(tensor):
    """Sincronizza e fa la media di un tensore/valore tra tutte le GPU."""
    if not dist.is_initialized():
        return tensor.item() if isinstance(tensor, torch.Tensor) else tensor
    
    # Se è un float nativo, lo trasformiamo in tensore per la comunicazione NCCL
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=torch.cuda.current_device())
    else:
        tensor = tensor.clone().detach()
        
    # All-Reduce calcola la somma del tensore su tutte le GPU
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # Dividiamo per il numero totale di GPU per ottenere la media globale
    tensor /= dist.get_world_size()
    return tensor.item()


def one_epoch(model, dataloader, optimizer, loss_fn, device, scaler, lambda_actor = 1.0, lambda_refiner = 1.0, validation = False):
    
    is_distributed = dist.is_initialized()
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    if validation:
        model.eval()
        grad_modality = torch.no_grad()
    else:
        model.train()
        grad_modality = torch.enable_grad()

    epoch_loss = 0 
    epoch_loss_actor = 0
    epoch_loss_refiner = 0
    epoch_mae_xyz = 0   
    epoch_mae_gripper = 0 
    epoch_cosim_ori = 0 

    # Mostriamo la barra tqdm SOLO sulla GPU principale (rank 0)
    pbar = tqdm(
        dataloader, 
        desc=f"Validation" if validation else "Training",
        disable=not is_main_process
    )  

    with grad_modality:
        for batch in pbar:
            
            vision_input = batch['vision_input'].to(device)
            text_input = batch['text_input'].to(device) if torch.is_tensor(batch['text_input']) else batch['text_input']
            joint_input = batch['joint_input'].to(device)
            action_seq_target = batch['action_seq_target'].to(device)
            
            with autocast(device_type='cuda', enabled=('cuda' in str(device))):
                actor_action_seq_pred, refiner_action_seq_pred = model(text_input, vision_input, joint_input)

                loss_actor = loss_fn(actor_action_seq_pred, action_seq_target)
                loss_refiner = loss_fn(refiner_action_seq_pred, action_seq_target)
                loss = (lambda_actor*loss_actor) + (lambda_refiner*loss_refiner)

            with torch.no_grad():
                mae_xyz = torch.abs(refiner_action_seq_pred[:, :, :3] - action_seq_target[:, :, :3]).mean()
                mae_grip = torch.abs(refiner_action_seq_pred[:, :, -1] - action_seq_target[:, :, -1]).mean()
                
                pred_ori = refiner_action_seq_pred[:, :, 3:6].reshape(-1, 3)
                target_ori = action_seq_target[:, :, 3:6].reshape(-1, 3)
                cosim_ori = F.cosine_similarity(pred_ori, target_ori, dim=-1).mean()
                        
            if not validation:
                optimizer.zero_grad()
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    # --- AGGIORNATO PER FSDP: Accesso alle proprietà tramite _fsdp_wrapped_module ---
                    policy_type = model._fsdp_wrapped_module.policy if is_distributed else model.policy
                    if policy_type == 'transformer':
                        # Sotto FSDP si chiama direttamente la funzione nativa del wrapper
                        model.clip_grad_norm_(max_norm=1.0)
        
                    scaler.step(optimizer)
                    scaler.update() 
                else:
                    loss.backward()
                    
                    # --- AGGIORNATO PER FSDP: Accesso alle proprietà tramite _fsdp_wrapped_module ---
                    policy_type = model._fsdp_wrapped_module.policy if is_distributed else model.policy
                    if policy_type == 'transformer':
                        model.clip_grad_norm_(max_norm=1.0)
                        
                    optimizer.step()
            
            epoch_loss += loss.item()
            epoch_loss_actor += loss_actor.item()
            epoch_loss_refiner += loss_refiner.item()
            epoch_mae_xyz += mae_xyz.item()
            epoch_mae_gripper += mae_grip.item()
            epoch_cosim_ori += cosim_ori.item()

            if is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'actor': f"{loss_actor.item():.4f}",
                    'refiner': f"{loss_refiner.item():.4f}",
                })
        
        # Calcolo della media locale per l'istanza corrente
        num_batches = len(dataloader)
        loss_epoch_avg = epoch_loss / num_batches
        loss_epoch_actor_avg = epoch_loss_actor / num_batches
        loss_epoch_refiner_avg = epoch_loss_refiner / num_batches
        epoch_mae_xyz = epoch_mae_xyz / num_batches
        epoch_mae_gripper = epoch_mae_gripper / num_batches
        epoch_cosim_ori = epoch_cosim_ori / num_batches

        # Sincronizzazione All-Reduce per calcolare le metriche globali reali
        metrics = {
            'loss': reduce_tensor(loss_epoch_avg),
            'loss_actor': reduce_tensor(loss_epoch_actor_avg),
            'loss_refiner': reduce_tensor(loss_epoch_refiner_avg),
            'mae_xyz': reduce_tensor(epoch_mae_xyz),
            'mae_gripper': reduce_tensor(epoch_mae_gripper),
            'cosim_ori': reduce_tensor(epoch_cosim_ori)
        }
    
    return metrics


def one_epoch_pred(predictor, vjepa_encoder, dataloader, optimizer, loss_fn, device, scaler, validation = False):
    
    is_distributed = dist.is_initialized()
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    if validation:
        predictor.eval()
        grad_modality = torch.no_grad()
    else:
        predictor.train()
        grad_modality = torch.enable_grad()

    epoch_loss = 0 
    
    pbar = tqdm(
        dataloader, 
        desc=f"Validation" if validation else "Training",
        disable=not is_main_process
    )  

    with grad_modality:
        for batch in pbar:
            
            frames_current = batch['frames_current'].to(device)
            frames_next = batch['frames_next'].to(device)
            action = batch['action'].to(device).unsqueeze(1)

            if scaler is not None or dist.is_initialized():
                vision_input = vision_input.half()
                if torch.is_tensor(joint_input):
                    joint_input = joint_input.half()

            with autocast(device_type='cuda', enabled=('cuda' in str(device))):
                with torch.no_grad():
                    z_obs_current = vjepa_encoder(frames_current)
                    z_obs_next = vjepa_encoder(frames_next)

                z_pred, _, _ = predictor(z_obs_current, action)
                loss = loss_fn(z_pred, z_obs_next)
                        
            if not validation:
                optimizer.zero_grad()
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update() 
                else:
                    loss.backward()
                    optimizer.step()
            
            epoch_loss += loss.item()
            
            if is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                })
        
        loss_epoch_avg = epoch_loss / len(dataloader)
        
        # All-Reduce anche per la loss del predictor
        metrics = {
            'loss': reduce_tensor(loss_epoch_avg),
        }
    
    return metrics