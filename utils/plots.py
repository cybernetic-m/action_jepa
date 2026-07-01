import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == "__main__":

    training_dir_path = "./results/results_alcor_9/2026_06_27__18_08"
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
    # IMMAGINE 1: LOSSES (Layout asimmetrico con GridSpec)
    # Riga 1: Total Loss, Predictor Loss
    # Riga 2: Actor, Refiner, Val Comparison
    # ==============================================================================
    fig1 = plt.figure(figsize=(18, 10))
    
    # Creiamo una griglia 2x6
    gs = gridspec.GridSpec(2, 6, figure=fig1)

    # Assegniamo gli spazi: 
    # Riga 1: 2 grafici da 3 colonne l'uno (centrati)
    ax_total = fig1.add_subplot(gs[0, 0:3])
    ax_pred = fig1.add_subplot(gs[0, 3:6])
    
    # Riga 2: 3 grafici da 2 colonne l'uno (distribuiti uniformemente)
    ax_actor = fig1.add_subplot(gs[1, 0:2])
    ax_ref = fig1.add_subplot(gs[1, 2:4])
    ax_val_comp = fig1.add_subplot(gs[1, 4:6])

    # 1. Total Loss (Prima riga, sinistra)
    ax_total.plot(df['Epoch'], df['loss_train'], label='Train', linewidth=2, color=c_train)
    ax_total.plot(df['Epoch'], df['loss_val'], label='Validation', linewidth=2, color=c_val)
    ax_total.set_title('Total Loss', fontweight='bold')
    ax_total.set_xlabel('Epoch')
    ax_total.set_ylabel('Loss')
    ax_total.grid(True, alpha=0.3)
    ax_total.legend()

    # 2. Predictor Loss (Prima riga, destra)
    if 'loss_predictor_train' in df.columns and 'loss_predictor_val' in df.columns:
        ax_pred.plot(df['Epoch'], df['loss_predictor_train'], label='Train', color=c_train)
        ax_pred.plot(df['Epoch'], df['loss_predictor_val'], label='Val', color=c_val)
        ax_pred.set_title('Predictor Loss', fontweight='bold')
        ax_pred.set_xlabel('Epoch')
        ax_pred.set_ylabel('Value')
        ax_pred.grid(True, alpha=0.3)
        ax_pred.legend()
    else:
        ax_pred.axis('off') # Se non ci sono i dati, lascia lo spazio vuoto in modo pulito

    # 3. Actor Loss (Seconda riga, sinistra)
    ax_actor.plot(df['Epoch'], df['loss_actor_train'], label='Train', color=c_train)
    ax_actor.plot(df['Epoch'], df['loss_actor_val'], label='Val', color=c_val)
    ax_actor.set_title('Actor Loss', fontweight='bold')
    ax_actor.set_xlabel('Epoch')
    ax_actor.set_ylabel('Value')
    ax_actor.grid(True, alpha=0.3)
    ax_actor.legend()

    # 4. Refiner Loss (Seconda riga, centro)
    ax_ref.plot(df['Epoch'], df['loss_refiner_train'], label='Train', color=c_train)
    ax_ref.plot(df['Epoch'], df['loss_refiner_val'], label='Val', color=c_val)
    ax_ref.set_title('Refiner Loss', fontweight='bold')
    ax_ref.set_xlabel('Epoch')
    ax_ref.set_ylabel('Value')
    ax_ref.grid(True, alpha=0.3)
    ax_ref.legend()

    # 5. Validation Loss Comparison (Seconda riga, destra)
    ax_val_comp.plot(df['Epoch'], df['loss_actor_val'], label='Actor Val', color=c_actor, linewidth=2.5)
    ax_val_comp.plot(df['Epoch'], df['loss_refiner_val'], label='Refiner Val', color=c_refiner, linewidth=2.5)
    ax_val_comp.set_title('Validation Loss Comparison', fontweight='bold')
    ax_val_comp.set_xlabel('Epoch')
    ax_val_comp.set_ylabel('Value')
    ax_val_comp.grid(True, alpha=0.3)
    ax_val_comp.legend()

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