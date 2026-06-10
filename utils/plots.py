import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    training_dir_path = "./results/results_cnr_3/2026_06_08__17_56"
    metrics_path = os.path.join(training_dir_path, 'metrics.csv')

    df = pd.read_csv(metrics_path)

    # Impostazione stile generale per paper accademici
    plt.style.use('seaborn-v0_8-muted')

    # Palette cromatica
    c_train = "#a72222"   # Rosso (Train generico)
    c_val = "#e8911e"     # Arancione (Validation generico)
    c_actor = "#5e3c99"   # Viola scuro (Actor)
    c_refiner = "#e7298a" # Rosa acceso (Refiner)
    c_lr = "#28b377"      # Verde (Learning Rate)
    c_pred = "#4a4a4a"    # Grigio scuro (Predictor)

    # ==============================================================================
    # IMMAGINE 1: LOSSES (4 plots: Total, Actor, Refiner, Val Comparison)
    # ==============================================================================
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes1 = axes1.flatten()

    # 1. Total Loss
    axes1[0].plot(df['Epoch'], df['loss_train'], label='Train', linewidth=2, color=c_train)
    axes1[0].plot(df['Epoch'], df['loss_val'], label='Validation', linewidth=2, color=c_val)
    axes1[0].set_title('Total Loss', fontweight='bold')
    axes1[0].set_xlabel('Epoch')
    axes1[0].set_ylabel('Loss')
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    # 2. Actor Loss
    axes1[1].plot(df['Epoch'], df['loss_actor_train'], label='Train', color=c_train)
    axes1[1].plot(df['Epoch'], df['loss_actor_val'], label='Val', color=c_val)
    axes1[1].set_title('Actor Loss', fontweight='bold')
    axes1[1].set_xlabel('Epoch')
    axes1[1].set_ylabel('Value')
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()

    # 3. Refiner Loss
    axes1[2].plot(df['Epoch'], df['loss_refiner_train'], label='Train', color=c_train)
    axes1[2].plot(df['Epoch'], df['loss_refiner_val'], label='Val', color=c_val)
    axes1[2].set_title('Refiner Loss', fontweight='bold')
    axes1[2].set_xlabel('Epoch')
    axes1[2].set_ylabel('Value')
    axes1[2].grid(True, alpha=0.3)
    axes1[2].legend()

    # 4. Validation Loss Comparison (Actor vs Refiner)
    axes1[3].plot(df['Epoch'], df['loss_actor_val'], label='Actor Val Loss', color=c_actor, linewidth=2.5)
    axes1[3].plot(df['Epoch'], df['loss_refiner_val'], label='Refiner Val Loss', color=c_refiner, linewidth=2.5)
    axes1[3].set_title('Validation Loss Comparison (Actor vs Refiner)', fontweight='bold')
    axes1[3].set_xlabel('Epoch')
    axes1[3].set_ylabel('Value')
    axes1[3].grid(True, alpha=0.3)
    axes1[3].legend()

    fig1.tight_layout()
    fig1.savefig(os.path.join(training_dir_path, 'plots_losses.png'), dpi=300)

    # ==============================================================================
    # IMMAGINE 2: METRICS (Effetto "Fading" sul Train, Focus sulla Validation)
    # ==============================================================================
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Parametri grafici per l'effetto Fading
    lw_train = 1.5
    alpha_train = 0.35  # Molto trasparente per fare da "sfondo"
    lw_val = 2.5
    alpha_val = 1.0     # Pieno e opaco per risaltare

    # 1. MAE XYZ
    axes2[0].plot(df['Epoch'], df['actor_mae_xyz_train'], label='Actor Train', color=c_actor, linewidth=lw_train, alpha=alpha_train)
    axes2[0].plot(df['Epoch'], df['refiner_mae_xyz_train'], label='Refiner Train', color=c_refiner, linewidth=lw_train, alpha=alpha_train)
    axes2[0].plot(df['Epoch'], df['actor_mae_xyz_val'], label='Actor Val', color=c_actor, linewidth=lw_val, alpha=alpha_val)
    axes2[0].plot(df['Epoch'], df['refiner_mae_xyz_val'], label='Refiner Val', color=c_refiner, linewidth=lw_val, alpha=alpha_val)
    axes2[0].set_title('MAE XYZ', fontweight='bold')
    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('Value')
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    # 2. MAE Gripper Error
    axes2[1].plot(df['Epoch'], df['actor_mae_gripper_train'], label='Actor Train', color=c_actor, linewidth=lw_train, alpha=alpha_train)
    axes2[1].plot(df['Epoch'], df['refiner_mae_gripper_train'], label='Refiner Train', color=c_refiner, linewidth=lw_train, alpha=alpha_train)
    axes2[1].plot(df['Epoch'], df['actor_mae_gripper_val'], label='Actor Val', color=c_actor, linewidth=lw_val, alpha=alpha_val)
    axes2[1].plot(df['Epoch'], df['refiner_mae_gripper_val'], label='Refiner Val', color=c_refiner, linewidth=lw_val, alpha=alpha_val)
    axes2[1].set_title('MAE Gripper Error', fontweight='bold')
    axes2[1].set_xlabel('Epoch')
    axes2[1].set_ylabel('Value')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    # 3. Cosine Similarity Orientation
    axes2[2].plot(df['Epoch'], df['actor_cosim_ori_train'], label='Actor Train', color=c_actor, linewidth=lw_train, alpha=alpha_train)
    axes2[2].plot(df['Epoch'], df['refiner_cosim_ori_train'], label='Refiner Train', color=c_refiner, linewidth=lw_train, alpha=alpha_train)
    axes2[2].plot(df['Epoch'], df['actor_cosim_ori_val'], label='Actor Val', color=c_actor, linewidth=lw_val, alpha=alpha_val)
    axes2[2].plot(df['Epoch'], df['refiner_cosim_ori_val'], label='Refiner Val', color=c_refiner, linewidth=lw_val, alpha=alpha_val)
    axes2[2].set_title('Cosine Similarity Orientation', fontweight='bold')
    axes2[2].set_xlabel('Epoch')
    axes2[2].set_ylabel('Value')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(training_dir_path, 'plots_metrics.png'), dpi=300)

    # ==============================================================================
    # IMMAGINE 3: HYPERPARAMS (2 plots: LR e Lambdas)
    # ==============================================================================
    fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # 1. Learning Rate
    axes3[0].plot(df['Epoch'], df['lr_train'], color=c_lr, linewidth=2.5)
    axes3[0].set_title('Learning Rate', fontweight='bold')
    axes3[0].set_xlabel('Epoch')
    axes3[0].set_ylabel('Value')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 2. Lambdas
    axes3[1].plot(df['Epoch'], df['lambda_actor_train'], label='Lambda Actor', color=c_actor, linewidth=2.5)
    axes3[1].plot(df['Epoch'], df['lambda_refiner_train'], label='Lambda Refiner', color=c_refiner, linewidth=2.5)
    
    # Controllo per il Predictor (con fallback a 1)
    if 'lambda_predictor_train' in df.columns:
        axes3[1].plot(df['Epoch'], df['lambda_predictor_train'], label='Lambda Predictor', color=c_pred, linewidth=2.5)
    else:
        axes3[1].plot(df['Epoch'], [1] * len(df['Epoch']), label='Lambda Predictor (Fixed=1)', color=c_pred, linewidth=2.5, linestyle='-')

    axes3[1].set_title('Loss Weights', fontweight='bold')
    axes3[1].set_xlabel('Epoch')
    axes3[1].set_ylabel('Value')
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend()

    fig3.tight_layout()
    fig3.savefig(os.path.join(training_dir_path, 'plots_hyperparams.png'), dpi=300)

    plt.show()