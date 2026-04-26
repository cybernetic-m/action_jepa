# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, "../"))

# Aggiunge la root e LIBERO al path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils import preprocess_libero_dataset
import torch
import json
import os
import cv2
from model.TransformerActionJEPA import TransformerActionJEPA

if __name__ == "__main__":
        
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    datasets_dir = "../LIBERO/libero/datasets"
    processed_data_dir ='../processed_data'
    
    with open('../config.json', 'r') as f:
        config = json.load(f)

    PREPROCESSING_WITH_BACKBONE = config['preprocessing_with_backbone'] 
    NUM_FRAMES = config['num_frames']
    DATASET_TYPE = config['dataset_type']
    INTERPOLATION_TYPE = config['interpolation_type']

    interpolation_dict= {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC
}

    # the get method return the value corresponding to the key INTERPOLATION TYPE, otherwise it return INTER CUBIC method
    interpolation = interpolation_dict.get(INTERPOLATION_TYPE, cv2.INTER_CUBIC)
    
    print(f"Preprocessing with backbone: {PREPROCESSING_WITH_BACKBONE}\nDataset: {DATASET_TYPE}\nInterpolation: {INTERPOLATION_TYPE}")

    if DATASET_TYPE == "all":
        selected_tasks = ["libero_10", "libero_90", "libero_spatial", "libero_goal", "libero_object"]
    else:
        selected_tasks = [DATASET_TYPE]
    
    libero_paths = [f"{datasets_dir}/{task}/*.hdf5" for task in selected_tasks]

    if PREPROCESSING_WITH_BACKBONE:
        checkpoints_path = "../checkpoints"
        # Path for all the models
        vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
        vjepa_pred_path = os.path.join(checkpoints_path,"facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar")
        clip_path = os.path.join(checkpoints_path,"openai/clip-vit-large-patch14")

        model = TransformerActionJEPA(
            vjepa_encoder_path=vjepa_path,
            vjepa_predictor_path=vjepa_pred_path,
            clip_model_path=clip_path,
            num_frames=NUM_FRAMES,
            use_backbone = True,
            device=device
        ).to(device)

    else:
        vision_backbone = None
        language_backbone = None
        
    for path in libero_paths:
        preprocess_libero_dataset(
                                hdf5_path=path,
                                output_dir=processed_data_dir, 
                                use_backbone = PREPROCESSING_WITH_BACKBONE,
                                model=model,
                                interpolation=interpolation
                                )


