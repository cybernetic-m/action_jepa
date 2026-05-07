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
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils import get_libero_path

if __name__ == "__main__":
        
    datasets_dir = "../LIBERO/libero/datasets"
    resample_data_dir = '../resampled_data'
    
    with open('../config.json', 'r') as f:
        config = json.load(f)

    DATASET_TYPE = config['dataset_type_preprocessing']

    if DATASET_TYPE == "all":
        selected_tasks = ["libero_spatial", "libero_goal", "libero_object", "libero_10", "libero_90"] 
    else:
        selected_tasks = [DATASET_TYPE]

    # PART OF RESAMPLING DATA 

    for dataset_name in selected_tasks:
        
        files = sorted(glob.glob(os.path.join(f"{datasets_dir}/{dataset_name}", "*.hdf5")))
        benchmark_dict = benchmark.get_benchmark_dict() 
        task_suite = benchmark_dict[dataset_name]() 
        all_task_names = task_suite.get_task_names()

        for file_path in files:
            task_name = os.path.basename(file_path).replace("_demo.hdf5","")
            task_id = all_task_names.index(task_name)

            data = sorted(glob.glob(os.path.join(f"{resample_data_dir}/{dataset_name}/{task_id}/data", "*.pt")))
            if len(data) < 50:

                print(f"\n[START] Starting resampling {dataset_name}\n")
                resample_data(
                            hdf5_path = file_path, 
                            task_id = task_id,
                            output_dir = resample_data_dir, 
                            task_suite_name = dataset_name)
            else:
                print(f"\n Task {task_id}: {os.path.basename(file_path)} just completed before!")

        print("\n[END] All tasks resampled")