import torch
import json
from training.one_epoch import one_epoch
from training.calculate_metrics import calculate_metrics

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, lambda_actor = 1.0, lambda_refiner = 1.0):

    # Initializing loss values (the best_vloss is very high at start to save surely the first model epoch)
    epoch_loss = 0
    epoch_val_loss = 0
    best_vloss = 100000000 

    # ADD METRICS
    ...

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch+1}/{num_epochs}")

        # One epoch of training 
        train_epoch_output = one_epoch(model, train_loader, optimizer, loss_fn, 
                                       device, lambda_actor, lambda_refiner
                                       )

        # Unpacking training output
        train_loss_avg, train_loss_act_avg, train_loss_ref_avg = train_epoch_output[:3]
        train_ref_action_target_list, train_ref_action_pred_list = train_epoch_output[3:5]
        train_act_action_seq_target_list, train_act_action_seq_pred_list = train_epoch_output[5:7]

        # One epoch of validation 
        val_epoch_output = one_epoch(model, val_loader, optimizer, loss_fn, 
                                       device, lambda_actor, lambda_refiner, 
                                       validation=True
                                       )

        # Unpacking validation output
        val_loss_avg, val_loss_act_avg, val_loss_ref_avg = val_epoch_output[:3]
        val_ref_action_target_list, val_ref_action_pred_list = val_epoch_output[3:5]
        val_act_action_seq_target_list, val_act_action_seq_pred_list = val_epoch_output[5:7]

        print(f"TRAIN\t Loss:{train_loss_avg}, Actor Loss:{train_loss_act_avg}, Refiner Loss:{train_loss_ref_avg}")
        print(f"VALIDATION\t Loss:{val_loss_avg}, Actor Loss:{val_loss_act_avg}, Refiner Loss:{val_loss_ref_avg}")