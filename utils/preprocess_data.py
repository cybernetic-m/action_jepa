# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, "../"))

model_path = os.path.join(root_path, "model")

if model_path not in sys.path:
    sys.path.insert(0, model_path)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import preprocess_data
import torch
import json
import os
from modules.CLIPEncoder import CLIPEncoder
from modules.VJEPAEncoder import VJEPAEncoder
import glob

if __name__ == "__main__":
        
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    datasets_dir = "../LIBERO/libero/datasets"
    processed_data_dir ='../processed_data'
    resample_data_dir = '../resampled_data'
    
    with open('../config.json', 'r') as f:
        config = json.load(f)

    NUM_FRAMES = config['num_frames']
    DATASET_TYPE = config['dataset_type_preprocessing']

    if DATASET_TYPE == "all":
        selected_tasks = ["libero_spatial", "libero_goal", "libero_object"]#, "libero_10", "libero_90"]
    else:
        selected_tasks = [DATASET_TYPE]

    # PART OF EXTRACT FEATURES FOR DATA
    
    # A list of the type: ['../resampled_data/libero_goal/2', '../resampled_data/libero_goal/3', '../resampled_data/libero_goal/9' ...]
    data_paths = sorted([f"{d}" for task in selected_tasks for d in glob.glob(os.path.join(resample_data_dir, task, "*")) if os.path.isdir(d)])
    
    checkpoints_path = "../checkpoints"
    # Path for all the models
    vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
    clip_path = os.path.join(checkpoints_path,"openai/clip-vit-large-patch14")

    vision_backbone = VJEPAEncoder(model_path=vjepa_path, frozen=True, device=device)
    language_backbone = CLIPEncoder(model_path=clip_path, frozen=True, device=device)
    
    
    for path in data_paths:
        
        preprocess_data(
                                data_dir=path,
                                output_dir=processed_data_dir, 
                                vision_backbone = vision_backbone,
                                language_backbone = language_backbone,
                                num_frames = NUM_FRAMES,
                                action_dim = 7,
                                )



