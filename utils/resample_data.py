# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, "../"))
libero_path = os.path.join(root_path, "LIBERO")

if libero_path not in sys.path:
    sys.path.insert(0, libero_path)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import resample_data
import json
import os
import glob

if __name__ == "__main__":
        
    datasets_dir = "../LIBERO/libero/datasets"
    resample_data_dir = '../resampled_data'
    
    with open('../config.json', 'r') as f:
        config = json.load(f)

    DATASET_TYPE = config['dataset_type_preprocessing']

    if DATASET_TYPE == "all":
        selected_tasks = ["libero_spatial", "libero_goal", "libero_object"]#, "libero_10", "libero_90"] 
    else:
        selected_tasks = [DATASET_TYPE]

    # PART OF RESAMPLING DATA 

    for dataset_name in selected_tasks:
        
        files = sorted(glob.glob(os.path.join(f"{datasets_dir}/{dataset_name}", "*.hdf5")))

        for i, file_path in enumerate(files):
            data = sorted(glob.glob(os.path.join(f"{resample_data_dir}/{dataset_name}/{i}/data", "*.pt")))
            if len(data) < 50:

                print(f"\n[START] Starting resampling\n Task {i}: {os.path.basename(file_path)}")
                resample_data(
                            hdf5_path = file_path, 
                            output_dir = resample_data_dir, 
                            task_id = i, 
                            task_suite_name = dataset_name)
            else:
                print(f"\n Task {i}: {os.path.basename(file_path)} just completed before!")

        print("\n[END] All tasks resampled")