import numpy as np
import torch
import cv2

def process_data(episode, num_frames, fps, window_second_size, lang_keys):
  # Function that having

  # Take the entire episode steps and load the single episode in RAM
  steps = list(episode['steps'])

  # Create lists of video_frames and states 
  video_frames = []
  states = []

  # Take all language instructions and create a list with all the texts
  language_instructions = [steps[0][key].numpy().decode('utf-8') for key in lang_keys]

  num_steps = len(steps) # num of total steps of the episode (we have 1 frame RGB per step)
  window_steps = window_second_size * fps # compute how much steps in our window we take

  # If the video is less then 4s we skip the video returning None 
  if num_steps < window_steps:
    return None, None, None, None
  
  # Otherwise we'll take the last 4s part of the episode, taking num_frames frames!
  start_idx = num_steps - window_steps
  end_idx = num_steps-1
  indices = np.linspace(start_idx, end_idx, num_frames).astype(int)

  for idx in indices:
    # Extract the image at idx, transform to numpy, resize to 256x256 with opencv library and append to video_frames list
    # 'img' is a 3D NumPy array with shape (256, 256, 3) representing [Height, Width, RGB Channels]:
    # [
    #   [[R, G, B], [R, G, B], ...],  <- Row 1: contains 256 pixels (each pixel is a 3-value vector)
    #   [[R, G, B], [R, G, B], ...],  <- Row 2: contains 256 pixels
    #   ...
    #   [[R, G, B], [R, G, B], ...]   <- Row 256
    # ]
    # Inner values (0-255) define the intensity for Red, Green, and Blue respectively.  
    img = steps[idx]['observation']['exterior_image_1_left'].numpy()
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # used to save the color sequence correctly
    video_frames.append(img) 

    # Extract the state (end effector pos+orientation+gripper => 7D state dim) at idx
    # 'state' is a 1D NumPy array with shape (7,) representing the robot's configuration:
    # [
    #   X, Y, Z,        <- End-Effector (EEF) Cartesian position
    #   Roll, Pitch, Yaw, <- EEF Orientation (Euler angles)
    #   Gripper         <- 1D Gripper state (Open/Close value)
    # ]
    # It is formed by concatenating the 6D 'cartesian_position' and the 1D 'gripper_position'.
    state_pr = steps[idx]['observation']['cartesian_position'].numpy() # 6D position and orientation of the EEF
    state_gripper = steps[idx]['observation']['gripper_position'].numpy() # 1D gripper position
    state = np.concatenate([state_pr, state_gripper])
    states.append(state)

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
  
  return video_pt, states_pt, action_pt, language_instructions


    
   
      
      
      

    
      


    