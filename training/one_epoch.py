from tqdm import tqdm
import torch

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

    # Lists of the refiner target and pred actions
    refiner_action_target_list = []
    refiner_action_pred_list = []

    # Lists of the actor target and pred actions
    actor_action_seq_target_list = []
    actor_action_seq_pred_list = []  

    pbar = tqdm(dataloader, desc=f"Validation" if validation else "Training")  

    # Iterating in batches
    with grad_modality:
        for batch in pbar:
            
            # Taking the data from the batch
            frames = batch['vision_input'].to(device)
            text_instruction = batch['text_input'].to(device) if torch.is_tensor(batch['text_input']) else batch['text_input']
            actor_action_seq_target = batch['actor_action_seq_target'].to(device)
            refiner_action_target = batch['refiner_action_target'].to(device)

            # Appending to lists the target values
            actor_action_seq_target_list.append(actor_action_seq_target.detach().cpu())
            refiner_action_target_list.append(refiner_action_target.detach().cpu())

            # Making the predictions
            actor_action_seq_pred, refiner_action_pred = model(text_instruction, frames)

            # Appending to lists the predicted values
            actor_action_seq_pred_list.append(actor_action_seq_pred.detach().cpu())
            refiner_action_pred_list.append(refiner_action_pred.detach().cpu())

            # Calculate the loss (the loss is a weighted sum of the actor loss and refiner loss)
            loss_actor = loss_fn(actor_action_seq_pred, actor_action_seq_target)
            loss_refiner = loss_fn(refiner_action_pred, refiner_action_target)
            loss = (lambda_actor*loss_actor) + (lambda_refiner*loss_refiner)
                        
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

            # Updating values in the bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'actor loss': f"{loss_actor.item():.4f}",
                'refiner loss': f"{loss_refiner.item():.4f}"
            })
        
        loss_epoch_avg = epoch_loss / len(dataloader)
        loss_epoch_actor_avg = epoch_loss_actor / len(dataloader)
        loss_epoch_refiner_avg = epoch_loss_refiner / len(dataloader)
    
    return (loss_epoch_avg, loss_epoch_actor_avg, loss_epoch_refiner_avg, refiner_action_target_list, refiner_action_pred_list, actor_action_seq_target_list, actor_action_seq_pred_list)

            
