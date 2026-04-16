import torch
import json
from training.one_epoch import one_epoch
import os
from datetime import datetime
import pandas as pd

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, config, results_dir_path, lambda_actor = 1.0, lambda_refiner = 1.0):
    
    # time stamp for creating a directory of the type ./results/2026_04_16__15_45
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
    training_dir_path = os.path.join(results_dir_path, timestamp)
    os.makedirs(training_dir_path, exist_ok = True)

    # Initializing loss value (the best_vloss is very high at start to save surely the first model epoch)
    best_vloss = 100000000 
    best_epoch = 1

    config_dict = {
        'hyperparameters': config,
    }

    json_save_path = os.path.join(training_dir_path, "experiment_config.json")
    with open(json_save_path, "w") as f:
        json.dump(config_dict, f, indent=4)


    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        print(f"\n" + "="*30)
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        print("="*30)

        # One epoch of training 
        train_metrics = one_epoch(model, train_loader, optimizer, loss_fn, 
                                       device, lambda_actor, lambda_refiner
                                       )

        # One epoch of validation 
        val_metrics = one_epoch(model, val_loader, optimizer, loss_fn, 
                                       device, lambda_actor, lambda_refiner, 
                                       validation=True
                                       )
        
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print("-" * 80)

        print(f"TRAIN      | Tot Loss: {train_metrics['loss']:.4f} | "
              f"Actor: {train_metrics['loss_actor']:.4f} | "
              f"Refiner: {train_metrics['loss_refiner']:.4f}")

        print(f"VALIDATION | Tot Loss: {val_metrics['loss']:.4f} | "
              f"Actor: {val_metrics['loss_actor']:.4f} | "
              f"Refiner: {val_metrics['loss_refiner']:.4f}")
        
        print(f"METRICS  (Validation)  | XYZ Err: {val_metrics['mae_xyz']:.2f} | "
              f"Cosine Sim Orientation: {val_metrics['cosim_ori']:.3f}")
        
        print("-" * 80) 

        current_vloss = val_metrics['loss']
        if current_vloss < best_vloss:
            best_vloss = current_vloss
            best_epoch = epoch+1
            model_save_path = os.path.join(training_dir_path, f"best_model.pth")
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
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
       
        
        