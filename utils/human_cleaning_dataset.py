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
import numpy as np

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

    cv2.namedWindow("Robot View")
    cv2.namedWindow("User Choice")

    cv2.moveWindow("Robot View", 100, 100) 
    cv2.moveWindow("User Choice", 620, 100)

    cv2.setMouseCallback("User Choice", mouse_callback)
    
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

                    current_idx = demo_files.index(demo) + 1 
                    remaining = total_demos - current_idx

                    # 1. FINESTRA ROBOT (PULITA)
                    robot_display = cv2.resize(frame, (1024, 1024))
                    cv2.imshow("Robot View", robot_display)

                    # 2. FINESTRA INFO
                    info_display = np.full((300, 512, 3), (45, 45, 45), dtype=np.uint8)
                                    
                    # Demo Name e Progresso
                    cv2.putText(info_display, f"Demo: {demo_name}", (20, 110), 1, 0.9, (200, 200, 200), 1)
                    cv2.putText(info_display, f"Progress: {current_idx}/{total_demos} ({remaining} left)", (20, 135), 1, 0.9, (255, 255, 255), 1)
                    
                    # Stats
                    s_count = current_info['manual_cleaning_stats']['success']
                    f_count = current_info['manual_cleaning_stats']['fail']
                    cv2.putText(info_display, f"Total Success: {s_count}", (20, 170), 1, 1, (0, 255, 0), 1)
                    cv2.putText(info_display, f"Total Fail: {f_count}", (280, 170), 1, 1, (0, 0, 255), 1)

                    # Pulsanti
                    cv2.rectangle(info_display, (50, 200), (230, 260), (0, 150, 0), -1)
                    cv2.putText(info_display, "SUCCESS (S)", (70, 240), 1, 1.2, (255, 255, 255), 2)
                    
                    cv2.rectangle(info_display, (280, 200), (460, 260), (0, 0, 150), -1)
                    cv2.putText(info_display, "FAIL (F)", (325, 240), 1, 1.2, (255, 255, 255), 2)

                    cv2.imshow("User Choice", info_display)

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
    




