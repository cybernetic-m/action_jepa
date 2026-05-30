import os
import shutil
import json


def auto_cleaning_dataset(resampled_data_dir, json_dict_path):

    if not os.path.exists(json_dict_path):
        print(f"Error: File {json_dict_path} do not exists.")
        return

    with open(json_dict_path, 'r') as f:
        cleaning_results = json.load(f)

    dataset_name = os.path.basename(json_dict_path)
    dataset_name = dataset_name.replace('.json', '')

    print(f"Starting auto-cleaning dataset {dataset_name} of {len(cleaning_results)} demo...")
    fail_count = 0

    task_stats = {}

    for demo_key, status in cleaning_results.items():
        task_id = demo_key.split('_')[1]
        if task_id not in task_stats:
            task_stats[task_id] = {"success": 0, "fail": 0}
        if status == 'fail':
            # Parsing: task_0_demo_17 -> task_id = 0
            task_stats[task_id]["fail"] += 1
            demo_filename = f"{demo_key}.pt"

            # DATA directory path
            task_path = os.path.join(resampled_data_dir, dataset_name, task_id)
            src_path = os.path.join(task_path, 'data', demo_filename)
            
            # FAIL directory path
            fail_dir = os.path.join(task_path, 'fail')
            dest_path = os.path.join(fail_dir, demo_filename)

            if os.path.exists(src_path):
                os.makedirs(fail_dir, exist_ok=True)
                shutil.move(src_path, dest_path)
                print(f"FAIL: {demo_key}")
                fail_count += 1
        else:
            task_stats[task_id]["success"] += 1
                
    print("-" * 30)
    print(f"END: Fail demo counted: {fail_count}")

    # Save statistics into each task directory
    print("\nSaving cleaning statistics...")
    
    for task_id, stats in task_stats.items():
        task_path = os.path.join(resampled_data_dir, dataset_name, task_id)
        
        # Preventative check to ensure the task directory exists
        if not os.path.exists(task_path):
            continue

        # 1. Update info.json
        info_path = os.path.join(task_path, 'info.json')
        info_data = {}
        
        # If info.json already exists, load its content to avoid overwriting existing keys
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
            except Exception:
                info_data = {}
        
        # Add or update the automatic cleaning stats field
        info_data["automatic_cleaning_stats"] = stats
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=4)

    print("[SUCCESS] All stats files successfully updated and saved.")

if __name__ == "__main__":

    resampled_data_dir = '../resampled_data'
    json_paths = ['./libero_goal.json', './libero_10.json', './libero_object.json', './libero_spatial.json']
    
    for path in json_paths:
        auto_cleaning_dataset(resampled_data_dir, path)