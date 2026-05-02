# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, "../"))

libero_path = os.path.join(root_path, "LIBERO")
model_path = os.path.join(root_path, "model")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)
if model_path not in sys.path:
    sys.path.insert(0, model_path)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import preprocess_data, resample_data
import torch
import json
import os
#from model.modules.CLIPEncoder import CLIPEncoder
#from model.modules.VJEPAEncoder import VJEPAEncoder
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
    DATASET_TYPE = config['dataset_type']

    if DATASET_TYPE == "all":
        selected_tasks = ["libero_10", "libero_90", "libero_spatial", "libero_goal", "libero_object"]
    else:
        selected_tasks = [DATASET_TYPE]

    # PART OF RESAMPLING DATA 

    for dataset_name in selected_tasks:
        
        files = sorted(glob.glob(os.path.join(f"{datasets_dir}/{dataset_name}", "*.hdf5")))

        for i, file_path in enumerate(files):
            print(f"\n[START] Starting resampling\n Task {i}: {os.path.basename(file_path)}")
            resample_data(
                        hdf5_path = file_path, 
                        output_dir = resample_data_dir, 
                        task_id = i, 
                        task_suite_name = dataset_name)

        print("\n[END] All tasks resampled")
    
    
    # PART OF EXTRACT FEATURES FOR DATA

    libero_paths = [f"{datasets_dir}/{task}/*.hdf5" for task in selected_tasks]

    '''
    checkpoints_path = "../checkpoints"
    # Path for all the models
    vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
    clip_path = os.path.join(checkpoints_path,"openai/clip-vit-large-patch14")

    vision_backbone = VJEPAEncoder(model_path=vjepa_path, frozen=True, device=device)
    language_backbone = CLIPEncoder(model_path=clip_path, frozen=True, device=device)
    '''
    
    for path in libero_paths:

        '''
        preprocess_data(
                                hdf5_path=path,
                                output_dir=processed_data_dir, 
                                vision_backbone = vision_backbone,
                                language_backbone = language_backbone,
                                num_frames = NUM_FRAMES,
                                action_dim = 7,
                                )
        '''


