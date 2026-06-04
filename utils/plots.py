import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    training_dir_path = "./results/results_alcor_1/2026_05_31__22_42"
    metrics_path = os.path.join(training_dir_path, 'metrics.csv')

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 18))
    axes = axes.flatten()
    plt.style.use('seaborn-v0_8-muted')

    c_train = "#a72222"
    c_val = "#e8911e"

    axes[0].plot(df['Epoch'], df['loss_train'], label='Train', linewidth=2, color=c_train)
    axes[0].plot(df['Epoch'], df['loss_val'], label='Validation', linewidth=2, color=c_val)
    axes[0].set_title('Total Loss', fontweight='bold')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df['Epoch'], df['loss_actor_train'], label='Train', color=c_train)
    axes[1].plot(df['Epoch'], df['loss_actor_val'], label='Val', color=c_val)
    axes[1].set_title('Actor Loss', fontweight='bold')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(df['Epoch'], df['loss_refiner_train'], label='Train', color=c_train)
    axes[2].plot(df['Epoch'], df['loss_refiner_val'], label='Val', color=c_val)
    axes[2].set_title('Refiner Loss', fontweight='bold')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(df['Epoch'], df['mae_xyz_train'], label='Train', color=c_train)
    axes[3].plot(df['Epoch'], df['mae_xyz_val'], label='Val', color=c_val)
    axes[3].set_title('MAE XYZ', fontweight='bold')
    axes[3].set_ylabel('Error [m]')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(df['Epoch'], df['mae_gripper_train'], label='Train', color=c_train)
    axes[4].plot(df['Epoch'], df['mae_gripper_val'], label='Val', color=c_val)
    axes[4].set_title('MAE Gripper Error', fontweight='bold')
    axes[4].set_ylabel('Error')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(df['Epoch'], df['cosim_ori_train'], label='Train', color=c_train)
    axes[5].plot(df['Epoch'], df['cosim_ori_val'], label='Val', color=c_val)
    axes[5].set_title('Cosine Similarity Orientation', fontweight='bold')
    axes[5].set_ylabel('Cosine Similarity Orientation')
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    axes[6].plot(df['Epoch'], df['lr_train'], color="#28b377", linewidth=2)
    axes[6].set_title('Learning Rate', fontweight='bold')
    axes[6].set_ylabel('LR')
    axes[6].grid(True, alpha=0.3)
    axes[6].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    axes[7].plot(df['Epoch'], df['loss_actor_val'], color="#fe9d0c", label='Loss Actor', linewidth=2)
    axes[7].plot(df['Epoch'], df['loss_refiner_val'], color="#d81a33", label='Loss Refiner', linewidth=2)
    axes[7].set_title('Validation Losses', fontweight='bold')
    axes[7].set_ylabel('Validation Losses')
    axes[7].grid(True, alpha=0.3)
    axes[7].ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    plt.tight_layout()
    plt.savefig(os.path.join(training_dir_path,'plots.png'), dpi=300)
    plt.show()