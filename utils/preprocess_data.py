# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import argparse
from utils.utils import preprocess_libero_dataset
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
import torch
import json

if __name__ == "__main__":

    datasets_dir = "./LIBERO/libero/datasets"
    processed_data_dir ='./processed_data'
    parser = argparse.ArgumentParser(description="Script to preprocess data")

    parser.add_argument('--dataset', type=str, default='all', help='Select the dataset to preprocess ["libero_goal", "libero_spatial", "libero_object", "libero_90", "libero_10"] (if you do not write anything "all" will be applied)')
    parser.add_argument('--use_backbone', action='store_true', help='Use V-JEPA 2 and CLIP backbones to preprocess data for faster training')

    args = parser.parse_args()
    print(f"Preprocessing with backbone: {args.use_backbone}\nDataset: {args.dataset}")

    with open('config.json', 'r') as f:
        config = json.load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if args.dataset == "all":
        selected_tasks = ["libero_10", "libero_90", "libero_spatial", "libero_goal", "libero_object"]
    else:
        selected_tasks = [args.dataset]
    
    libero_paths = [f"{datasets_dir}/{task}/*.hdf5" for task in selected_tasks]

    if args.use_backbone:
        # Path for the VJEPA Vision Encoder
        vjepa_path = "checkpoints/facebook/vjepa2-vitg-fpc64-256"

        vision_backbone = VJEPAEncoder(
            model_path=vjepa_path,
            frozen=True,
            device=device
        ).to(device)

        # Path for the CLIP Language Encoder 
        clip_path = "checkpoints/openai/clip-vit-large-patch14"

        language_backbone = CLIPEncoder(
            model_path=clip_path,
            max_length = config['max_length'],
            frozen=True,
            device=device
        ).to(device)
    else:
        vision_backbone = None
        language_backbone = None
        
    for path in libero_paths:
        preprocess_libero_dataset(hdf5_path=path,
                                output_dir=processed_data_dir,
                                vision_backbone = vision_backbone, 
                                language_backbone = language_backbone, 
                                use_backbone = args.use_backbone
                                )


