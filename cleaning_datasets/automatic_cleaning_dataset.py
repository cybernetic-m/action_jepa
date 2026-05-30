import os
import shutil
import json
import glob

def auto_cleaning_dataset(resampled_data_dir, json_dict_path):

    if not os.path.exists(json_dict_path):
        print(f"Error: File {json_dict_path} do not exists.")
        return

    with open(json_dict_path, 'r') as f:
        cleaning_results = json.load(f)

    print(f"Starting auto-cleaning of {len(cleaning_results)} demo...")
    fail_count = 0

    for demo_key, status in cleaning_results.items():
        if status == 'fail':
            # Parsing: task_0_demo_17 -> task_id = 0
            task_id = demo_key.split('_')[1]
            demo_filename = f"{demo_key}.pt"

            # DATA directory path
            task_path = os.path.join(resampled_data_dir, "libero_goal", task_id)
            src_path = os.path.join(task_path, 'data', demo_filename)
            
            # FAIL directory path
            fail_dir = os.path.join(task_path, 'fail')
            dest_path = os.path.join(fail_dir, demo_filename)

            if os.path.exists(src_path):
                os.makedirs(fail_dir, exist_ok=True)
                shutil.move(src_path, dest_path)
                print(f"FAIL: {demo_key}")
                fail_count += 1
                
    print("-" * 30)
    print(f"END: Fail demo counted: {fail_count}")

if __name__ == "__main__":

    resampled_data_dir = '../resampled_data'
    json_paths = ['./libero_goal.json', './libero_10.json', './libero_object.json', './libero_spatial.json']
    
    for path in json_paths:
        auto_cleaning_dataset(resampled_data_dir, path)