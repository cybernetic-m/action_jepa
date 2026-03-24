import numpy as np
import torch
import cv2
import h5py
from tqdm import tqdm
import glob
import json
import os


def preprocess_libero_dataset(hdf5_path, output_dir):
   
   # Create the output directory if it does not exist where to save the .pt file
   os.makedirs(output_dir)

   # List of all the files hdf5 in the path ['./path_to_file1/file1.hdf5', ....]
   files = glob.glob(hdf5_path)

   # Iterating in all the file paths: each file is formed by different "demo" with a demo_id: 'demo_1', 'demo_2' ...
   for file in files:
    with h5py.File(file, 'r') as f:
        for demo_id in f['data'].keys():
          demo = f['data'][demo_id]

          # Take all the frames in the demo 
          frames = demo['obs']['agentview_rgb'][:]  # (T, H, W, 3)

          # Save the values T (time, num of frames), H (Height), W (Width), C (Channels)
          T, H, W, C = frames.shape
          
          # problem_info is a dict of the type .. that contain 'language_instruction'
          problem_info = json.loads(demo.attrs['problem_info'])
          text_instruction = problem_info['language_instruction'] # the text instruction

          # Flipping all the frames vertically because the original ones are flipped
          frames_flipped = np.flip(frames[::], axis=1)
          
          # Resize the frames from original size 128x128 to 256x256 size
          new_size = (256, 256)
          frames_resized = [] 

          for t in range(len(frames_flipped)):
              frame = frames_flipped[t]
              resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
              frames_resized.append(resized_frame)

        


def process_data(episode, num_frames, fps, window_second_size, lang_keys, dataset):
  # Function that having

  # Take the entire episode steps and load the single episode in RAM
  #steps = list(episode['steps'])

  num_steps = 0
  for _ in episode['steps']:
    num_steps += 1

  window_steps = window_second_size * fps # compute how much steps in our window we take

  # If the video is less then tot seconds we skip the video returning None 
  if num_steps < window_steps:
    return None, None, None, None, None, None
  
  # Otherwise we'll take the tot seconds videos in the middle part of the episode to try to catch the most "semantic" important motion
  # Compute the middle part of the episode (Ex. ep of 300 steps => ep_mid_idx = 150)
  ep_mid_idx = num_steps // 2
  # The window is centered at ep_mid_idx, it means we take the half of the window size (window_steps//2) and subtract this to the mid idx
  start_idx = ep_mid_idx - (window_steps//2)
  start_idx = max(0,start_idx) # if start_idx is negative, then take the first frame
  # The end_idx is simply start plus the window_steps size
  end_idx = start_idx + window_steps

  # Check if the end index go outside the total number of steps
  if end_idx > num_steps:
    end_idx = num_steps - 1 # take the last frame
    start_idx = max(0, end_idx - window_steps) # the start is 0 if end_idx - window_steps is negative

  indices = np.linspace(start_idx, end_idx, num_frames).astype(int)
  set_indices = set(indices)
  
  video_extracted_seconds = (indices[-1] - indices[0])/fps

  # Create lists of video_frames and states 
  video_frames = []
  states = []

  lang_dict = {key: "" for key in lang_keys}
  found_instruction = False

  '''
  if dataset == 'droid':
    # Take all language instructions and create a list with all the texts
    language_instructions = [steps[0][key].numpy().decode('utf-8') for key in lang_keys]
    print(language_instructions)
  elif dataset == 'bridge':
    language_instructions = [steps[0]['observation'][key].numpy().decode('utf-8') for key in lang_keys]
'''

  
  #num_steps = len(steps) # num of total steps of the episode (we have 1 frame RGB per step)
  

  for i, step in enumerate(episode['steps']):

    for key in lang_keys:
            if lang_dict[key] == "": 
                if dataset == "droid":
                    string = step[key].numpy().decode('utf-8').strip()
                else: # bridge
                    string = step['observation'][key].numpy().decode('utf-8').strip()
                
                if string:
                    lang_dict[key] = string
                    found_instruction = True

    # Extract the image at idx, transform to numpy, resize to 256x256 with opencv library and append to video_frames list
    # 'img' is a 3D NumPy array with shape (256, 256, 3) representing [Height, Width, RGB Channels]:
    # [
    #   [[R, G, B], [R, G, B], ...],  <- Row 1: contains 256 pixels (each pixel is a 3-value vector)
    #   [[R, G, B], [R, G, B], ...],  <- Row 2: contains 256 pixels
    #   ...
    #   [[R, G, B], [R, G, B], ...]   <- Row 256
    # ]
    # Inner values (0-255) define the intensity for Red, Green, and Blue respectively.  
    if i in set_indices:
      if dataset == 'droid':
        img = step['observation']['exterior_image_1_left'].numpy()
      elif dataset == 'bridge':
        img = step['observation']['image'].numpy()

      img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
      video_frames.append(img) 

      # Extract the state (end effector pos+orientation+gripper => 7D state dim) at idx
      # 'state' is a 1D NumPy array with shape (7,) representing the robot's configuration:
      # [
      #   X, Y, Z,        <- End-Effector (EEF) Cartesian position
      #   Roll, Pitch, Yaw, <- EEF Orientation (Euler angles)
      #   Gripper         <- 1D Gripper state (Open/Close value)
      # ]
      # It is formed by concatenating the 6D 'cartesian_position' and the 1D 'gripper_position'.
      if dataset == 'droid':
        state_pr = step['observation']['cartesian_position'].numpy() # 6D position and orientation of the EEF
        state_gripper = step['observation']['gripper_position'].numpy() # 1D gripper position
        state = np.concatenate([state_pr, state_gripper])
      elif dataset == 'bridge':
        state = step['observation']['state'].numpy()

      states.append(state)

  language_instructions = [lang_dict[key] for key in lang_keys]

  # Save the video frames in torch tensor
  # 1. np.stack: Converts the Python List of sixteen (256, 256, 3) arrays into a 
  #    single 4D NumPy array shaped (16, 256, 256, 3) [Frames, Height, Width, Channels].
  # 2. .permute(0, 3, 1, 2): Rearranges the dimensions to (16, 3, 256, 256) 
  #    [Time, Channels, Height, Width] to match the PyTorch 'Channels-First' standard.
  # 3. .byte(): Saves each pixel value (0-255) as a 1-byte unsigned integer (uint8).
  #    This reduces storage/RAM usage compared to float32 without losing data.
  video_pt = torch.from_numpy(np.stack(video_frames)).permute(0, 3, 1, 2).byte() # [16, 3, 256, 256]
  
  # Save the states in torch tensor 
  # 1. np.stack: Converts the Python List of sixteen (7,) arrays into a 
  #    single 2D NumPy array shaped (16, 7) [Frames, State].
  # Before np.stack: [array(7,), array(7,), ...] -> A list of 16 individual 7D vectors
  # After np.stack:  array(16, 7) -> A single 2D matrix where:
  #                  [ 
  #                    [x,y,z,r,p,y,g], -> 7D State of frame 1
  #                    [x,y,z,r,p,y,g], -> 7D State of frame 2
  #                    ...
  #                    [x,y,z,r,p,y,g]  -> 7D State of frame 16
  #                  ]
  # 2. .float(): Saves each state value as a float32 number.
  states_pt = torch.from_numpy(np.stack(states)).float()
  
  # We compute the actions as state_k+1 - state_k (next state minus the actual state). To do this:
  # 1. Compute states_pt[1:]: take the tensor [s_1, s_2, ..., s_15] where s_i is 7D tensor
  # 2. Compute states_pt[:-1]: take the tensor [s_0, s_1, ..., s_14] where s_i is 7D tensor
  # 3. Do the subtraction tensor by tensor [s_1-s_0, s_2-s_1, ..., s_15-s_14] resulting in a tensor [15,7]
  action_pt = states_pt[1:] - states_pt[:-1]

  # Normalize angle differences between [0,2pi] to [-pi,pi]. 
  # In fact before this if we have r1=3.14 (pi) and r0 = -3.14 (-pi) (the robot really do not move)
  # a = 6.28 means a full rotation of 2_pi, but the robot between frames do not move really
  # With normalization a = atan2(0,1) = 0
  action_pt[:, 3:6] = torch.atan2(torch.sin(action_pt[:, 3:6]), torch.cos(action_pt[:, 3:6]))
  
  return video_pt, states_pt, action_pt, language_instructions, found_instruction, video_extracted_seconds


def dataset2path(dataset_name):
  if dataset_name == 'droid':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'


   
   

    
   
      
      
      

    
      


    