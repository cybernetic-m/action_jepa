from tqdm import tqdm
import torch
import torch.nn.functional as F

def one_epoch(model, dataloader, optimizer, loss_fn, device, lambda_actor = 1.0, lambda_refiner = 1.0, validation = False):
    
    # Set the model in validation or training mode
    # Compute the gradient only if we are in training mode, 
    # in validation mode no gradient computation
    if validation:
        model.eval()
        grad_modality = torch.no_grad()
    else:
        model.train()
        grad_modality = torch.enable_grad()

    # Loss for this epoch reset to zero
    epoch_loss = 0 
    epoch_loss_actor = 0
    epoch_loss_refiner = 0
    epoch_mae_xyz = 0   # it's the mean absolute error for the position xyz of the gripper
    epoch_mae_gripper = 0 # it's the mean absolute error for the gripper [0;2] (gripper are in range -1,1)
    epoch_cosim_ori = 0 # it's the cosine similarity between the orientation vectors target and predicted

    pbar = tqdm(dataloader, desc=f"Validation" if validation else "Training")  

    # Iterating in batches
    with grad_modality:
        for batch in pbar:
            
            # Taking the data from the batch
            vision_input = batch['vision_input'].to(device)
            text_input = batch['text_input'].to(device) if torch.is_tensor(batch['text_input']) else batch['text_input']
            joint_input = batch['joint_input'].to(device)
            action_seq_target = batch['action_seq_target'].to(device)

            # Making the predictions
            actor_action_seq_pred, refiner_action_seq_pred = model(text_input, vision_input, joint_input)

            # Calculate the loss (the loss is a weighted sum of the actor loss and refiner loss)
            loss_actor = loss_fn(actor_action_seq_pred, action_seq_target)
            loss_refiner = loss_fn(refiner_action_seq_pred, action_seq_target)
            loss = (lambda_actor*loss_actor) + (lambda_refiner*loss_refiner)

            with torch.no_grad():
                # MAE XYZ
                mae_xyz = torch.abs(refiner_action_seq_pred[:, :, :3] - action_seq_target[:, :, :3]).mean()
                
                # MAE Gripper
                mae_grip = torch.abs(refiner_action_seq_pred[:, :, -1] - action_seq_target[:, :, -1]).mean()
                
                # Cosine similarity
                pred_ori = refiner_action_seq_pred[:, :, 3:6].reshape(-1, 3)
                target_ori = action_seq_target[:, :, 3:6].reshape(-1, 3)
                cosim_ori = F.cosine_similarity(pred_ori, target_ori, dim=-1).mean()
                        
            if not validation:
                # Zeroing the gradient
                optimizer.zero_grad()

                # Compute the backward pass (gradient values)
                loss.backward()

                # Update weights
                optimizer.step()
            
            # Incrementing the loss of the epoch
            epoch_loss += loss.item()
            epoch_loss_actor += loss_actor.item()
            epoch_loss_refiner += loss_refiner.item()
            epoch_mae_xyz += mae_xyz.item()
            epoch_mae_gripper += mae_grip.item()
            epoch_cosim_ori += cosim_ori.item()

            # Updating values in the bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'actor loss': f"{loss_actor.item():.4f}",
                'refiner loss': f"{loss_refiner.item():.4f}"
            })
        
        loss_epoch_avg = epoch_loss / len(dataloader)
        loss_epoch_actor_avg = epoch_loss_actor / len(dataloader)
        loss_epoch_refiner_avg = epoch_loss_refiner / len(dataloader)
        epoch_mae_xyz = epoch_mae_xyz / len(dataloader)
        epoch_mae_gripper = epoch_mae_gripper / len(dataloader)
        epoch_cosim_ori = epoch_cosim_ori / len(dataloader)

        metrics = {
            'loss': loss_epoch_avg,
            'loss_actor': loss_epoch_actor_avg,
            'loss_refiner': loss_epoch_refiner_avg,
            'mae_xyz': epoch_mae_xyz,
            'mae_gripper': epoch_mae_gripper,
            'cosim_ori': epoch_cosim_ori

        }
    
    return metrics

            
