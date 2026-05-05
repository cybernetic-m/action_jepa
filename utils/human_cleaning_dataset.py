# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, "../"))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import glob
import cv2
import imageio
import shutil
import json
from utils import draw_text

user_choice = None

def mouse_callback(event, x, y, flags, param):
    global user_choice

    # event is true when you click the left button of the mouse
    if event==cv2.EVENT_LBUTTONDOWN:
        if 50<=x<=230 and 440<=y<=490:
            user_choice='success'
        elif 280<=x<=460 and 440<=y<=490:
            user_choice='fail'


def manual_cleaning_dataset(resampled_data_dir):
    global user_choice
    # this contain paths of the type [./resampled_data_dir/libero_goal/0, ...., ./resampled_data_dir/libero_spatial/1]
    task_paths = sorted(glob.glob(os.path.join(resampled_data_dir, "*", "*")))

    cv2.namedWindow("Dataset Reviewer")
    cv2.setMouseCallback("Dataset Reviewer", mouse_callback)
    
    for task_path in task_paths:

        history_path = os.path.join(task_path, 'cleaning.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {} # Structure {'demo_name': 'success'/'fail'}

        info_path = os.path.join(task_path, 'info.json')
        current_info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                current_info = json.load(f)
        if 'manual_cleaning_stats' not in current_info:
            current_info['manual_cleaning_stats'] = {'success': 0, 'fail': 0}

        text_instruction = current_info.get('text_instruction', "Goal: Task execution")
                    
        gif_task_path = os.path.join(task_path, 'gifs') # the path of the gifs to load
        # Creating a directory where to store fail demo
        fail_dir = os.path.join(task_path, 'fail')
        os.makedirs(fail_dir, exist_ok=True)

        # List of all the pt file demos
        demo_files = sorted(glob.glob(os.path.join(task_path, 'data', '*pt')))

        # Keeping trace in the UI of the remaining demo to control
        total_demos = len(demo_files)
        demo_reviewed = len(history)

        # Iterating over all the demo file pt, showing to the user the corresponding gif
        for demo in demo_files:
            demo_name = os.path.basename(demo).replace('.pt','') # take the last part file1.pt eliminating .pt
            
            # Jump this demo if previously was reviewed
            if demo_name in history:
                continue

            gif_path = os.path.join(gif_task_path, f'{demo_name}.gif') # take the file1.gif

            # Loading the gif and creating a list of frames to show with opencv
            video = imageio.mimread(gif_path)
            frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in video]

            user_choice = None
            print(f"Reviewing {demo_name}")

            while user_choice is None:
                for frame in frames:

                    current_idx = list(demo_files).index(demo) + 1 
                    remaining = total_demos - current_idx

                    display = cv2.resize(frame, (512, 512))
                    
                    cv2.rectangle(display, (0, 370), (512, 512), (50, 50, 50), -1)

                    s_count = current_info['manual_cleaning_stats']['success']
                    f_count = current_info['manual_cleaning_stats']['fail']

                    draw_text(
                        img = display,
                        text = f"Goal: {text_instruction}",
                        position = (15, 385),
                        font = cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale = 0.45,
                        color = (255, 255, 255),
                        color_border = (0, 0, 0),
                        thickness = 1,
                        max_width = 480
                    )

                    # Counter visualization
                    cv2.putText(display, f"Total Success: {s_count}", (50, 425), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display, f"Total Fail: {f_count}", (280, 425), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.putText(display, f"Demo {current_idx} of {total_demos} ({remaining} left)", (10, 400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # SUCCESS (Green)
                    cv2.rectangle(display, (50, 440), (230, 490), (0, 150, 0), -1)
                    cv2.putText(display, "Success (S)", (85, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # FAIL (Red)
                    cv2.rectangle(display, (280, 440), (460, 490), (0, 0, 150), -1)
                    cv2.putText(display, "Fail (F)", (300, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(display, f" {demo_name}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow("Dataset Reviewer", display)

                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('s'): 
                        user_choice = 'success'
                        print(f" -> SAVED as SUCCESS")
                    elif key == ord('f'): 
                        user_choice = 'fail'
                        print(f" -> MARKED as FAIL")
                    elif key == ord('q'): 
                        cv2.destroyAllWindows()
                        return 
                    
                    if user_choice is not None: break
            
            # Adding this demo to the history
            history[demo_name] = user_choice
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            if user_choice== 'success':
                current_info['manual_cleaning_stats']['success'] += 1
            elif user_choice=='fail':
                current_info['manual_cleaning_stats']['fail']+=1
                shutil.move(demo, os.path.join(fail_dir, f"{demo_name}.pt"))

            with open(info_path, 'w') as f:
                json.dump(current_info, f, indent=4)
        
        cv2.destroyAllWindows()



if __name__ == "__main__":
        
   
    resample_data_dir = '../resampled_data'

    manual_cleaning_dataset(
        resampled_data_dir=resample_data_dir
    )
    




