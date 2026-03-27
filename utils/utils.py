import numpy as np
import torch
import cv2
import h5py
import glob
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
import textwrap

mpl.rcParams['animation.embed_limit'] = 100.0

def preprocess_libero_dataset(hdf5_path, output_dir, interpolation = cv2.INTER_LINEAR):
   
   # Create the output directory if it does not exist where to save the .pt file
   os.makedirs(output_dir, exist_ok=True)

   # List of all the files hdf5 in the path ['./path_to_file1/file1.hdf5', ....]
   files = glob.glob(hdf5_path)

   if not files:
      raise FileNotFoundError(f"No file founded in {hdf5_path}. Please download the LIBERO Dataset through the command 'python benchmark_scripts/download_libero_datasets.py'")
   
   # Dictionary we will save in json to remember correspondances between tasks and name id that we save
   task_map = {}

   # Iterating in all the file paths: each file is formed by different "demo" with a demo_id: 'demo_1', 'demo_2' ...
   for task_idx, file_path in enumerate(files):
    
    # From the entire path name ./path_to_file1/file1.hdf5 take only the final part file1.hdf5 eliminating .hdf5 
    task_name = os.path.basename(file_path).replace(".hdf5","")
    task_map[task_idx] = task_name # saving the correspondances (ex. '0': 'KITCHEN_SCENE3_turn...')

    with h5py.File(file_path, 'r') as f:
        # problem_info is a dict of the type .. that contain 'language_instruction'
        problem_info = json.loads(f['data'].attrs['problem_info'])
        text_instruction = problem_info['language_instruction'] # the text instruction
        
        for demo_id in tqdm(f['data'].keys(), desc=f"{task_name}"):

          demo = f['data'][demo_id]

          # Take all the frames in the demo 
          frames = demo['obs']['agentview_rgb'][:]  # (T, H, W, 3)

          # Save the values T (time, num of frames), H (Height), W (Width), C (Channels)
          T, H, W, C = frames.shape
          
          # Taking the corresponding end effector cartesian position, orientation and gripper states
          ee_pos = demo['obs']['ee_pos'][:] # (ee_x, ee_y, ee_z) three cartesian values of the end effector position
          ee_ori = demo['obs']['ee_ori'][:] # (r_x, r_y, r_z) three values in AXIS-ANGLE representation of the end effector orientation
          
          # For the gripper we have originally two values (ee_g_left, ee_g_right), equal in module but different in sign: ex. (0.0360, -0.0356)
          # that represent the opening of the left and right part of the gripper with respect to the center
          # I take the difference and divide by 2: (ee_g_left - ee_g_right) / 2 => (0.0360 - (-0.0356))/2 = 0.0715/2 = 0.0358
          ee_gripper_two_val = demo['obs']['gripper_states'][:] 
          ee_gripper = (ee_gripper_two_val[:,0] - ee_gripper_two_val[:,-1])/2
          ee_gripper = ee_gripper.reshape(-1, 1)
          ee_states = np.concatenate([ee_pos, ee_ori, ee_gripper], axis=1)

          # Flipping all the frames vertically because the original ones are flipped
          frames_flipped = np.flip(frames[::], axis=1)
          
          # Resize the frames from original size 128x128 to 256x256 size
          new_size = (256, 256)
          frames_resized = [] 

          for t in range(len(frames_flipped)):
              frame = frames_flipped[t]
              resized_frame = cv2.resize(frame, new_size, interpolation=interpolation)
              frames_resized.append(resized_frame)
          
          data = {"frames": torch.from_numpy(np.array(frames_resized)).byte(),
                  "text_instruction": text_instruction,
                  "ee_states": torch.from_numpy(ee_states).float()
                  }
          
          # saving something like task_0_demo_1.pt, you can see from task_map.json file that '0' is 'KITCHEN_SCENE....'
          save_name = f"task_{task_idx}_{demo_id}.pt"
          torch.save(data, os.path.join(output_dir, save_name))
   
   # Saving the correspondance map in json file
   with open(os.path.join(output_dir, 'task_map.json'), 'w') as f:
    json.dump(task_map, f)

def demo_animator(demo_pt_path):
  
  # Load the demo from the path as a dictionary
  demo = torch.load(demo_pt_path)

  # Load frames, text and end effector states tensors and transform in numpy
  frames = demo["frames"].numpy() # (steps, 256,256,3)
  text_instruction = demo["text_instruction"] # is text
  ee_states = demo["ee_states"] # (steps, 7)

  # Extract traj (trajectory), ori (axis-angle orientation), gripper (gripper open/close values)
  traj = ee_states[:, :3]
  ori = ee_states[:, 3:6]
  gripper = ee_states[:, 6]

  # Normalize values of gripper between [0,1] only for visualization of a bar plot
  gripper_norm = (gripper - gripper.min())/(gripper.max() - gripper.min()) 

  # Create a plot with 2x2 cells
  fig = plt.figure(figsize=(16,7))
  grid = fig.add_gridspec(2,2, width_ratios=[1,1], height_ratios=[1, 0.1])

  # Create the subplots 
  # upper left: the video gif
  # upper right: a 3D plot of the trajectory and orientation (as a ref frame that rotate) of the end effector
  # lower right: a bar plot to indicate the opening and closing of the end effector
  ax_video = fig.add_subplot(grid[0,0])
  ax_3dplot = fig.add_subplot(grid[0,1], projection='3d') # projection 3d means not a flat plot but 3d plot
  ax_gripper = fig.add_subplot(grid[1,1]) # this is an horizontal bar under the 3d plot to indicate the opening and closing of the gripper

  # Initialization of the ax_video
  ax_video.axis("off") # for the video we do not show axis
  im = ax_video.imshow(frames[0]) # initialization with first frame
  instruction_wrapped = "\n".join(textwrap.wrap(f"Instruction: {text_instruction}", width=50))
  ax_video.set_title(f"{instruction_wrapped}", fontsize=12) # Show as title the text instruction

  # Initialization of the ax_3dplot
  # q_x, q_y, q_z are the three vectors of the reference frame
  ax_3dplot.plot(traj[:,0], traj[:,1], traj[:,2], 'k--', alpha=0.8, zorder=1) # scatter plot of point in the trajectory
  q_x = ax_3dplot.quiver(X=0, Y=0, Z=0, U=0, V=0, W=0, color='r', length=0.05, zorder=5)  # x,y,z are the point of origin of the vector, u,v,w are the components of the vector
  q_y = ax_3dplot.quiver(X=0, Y=0, Z=0, U=0, V=0, W=0, color='g', length=0.05, zorder=5)
  q_z = ax_3dplot.quiver(X=0, Y=0, Z=0, U=0, V=0, W=0, color='b', length=0.05, zorder=5)

  ax_3dplot.view_init(elev=15, azim=45) # orientation of the 3D view
  ax_3dplot.set_xlabel('X [m]') # setting labels X, Y, Z for all axis
  ax_3dplot.set_ylabel('Y [m]')
  ax_3dplot.set_zlabel('Z [m]')

  # Select indices of steps where the gripper change the state (from Open (1) -> Closed (0) or viceversa)
  # change_state is a list formed by 0 where the robot gripper do not change the state or +1 (Open->Closed) and -1 (Closed -> Open)
  is_closed = gripper_norm < 0.7  # List of boolean False,True depending if the value is greater or less of 0.7
  change_state = np.diff(is_closed.int()) # transforming False = 0 and True = 1 and doing differences of element at next position minus element at current position
  indices_open = np.where(change_state == -1)[0] # returning the indices where we have -1 (from Closed -> Open)
  indices_closed = np.where(change_state == 1)[0] # returning the indices where we have +1 (from Open -> Closed)

  # Create four invisible line with color r, g, b and labels for the legend of orientaton axis and orange for the gripper closed legend
  line_x = ax_3dplot.plot([], [], [], color='r', label='X-axis')[0]
  line_y = ax_3dplot.plot([], [], [], color='g', label='Y-axis')[0]
  line_z = ax_3dplot.plot([], [], [], color='b', label='Z-axis')[0]
  line_gripper_closed = ax_3dplot.plot([], [], [], 'o', color='magenta', label='Gripper open Point')[0]
  line_gripper_open = ax_3dplot.plot([], [], [], 'o', color='cyan', label='Gripper closed Point')[0]

  # legend take all elements with label (the three lines) and add to a legend block upper right
  ax_3dplot.legend(loc='upper right', fontsize=10)

  # Imposta il rapporto tra gli assi a 1:1:1
  ax_3dplot.set_box_aspect([1,1,1])

  # Draw in the trajectory all the point in which the gripper is closed
  if len(indices_open) > 0:
      ax_3dplot.scatter(traj[indices_open, 0], 
                        traj[indices_open, 1], 
                        traj[indices_open, 2], 
                        color='magenta', s=50, label='Gripper Open', zorder=5)
      
  if len(indices_closed) > 0:
    ax_3dplot.scatter(traj[indices_closed, 0], 
                      traj[indices_closed, 1], 
                      traj[indices_closed, 2], 
                      color='cyan', s=50, label='Gripper Closed', zorder=5)


  # Initialization of the ax_gripper
  ax_gripper.set_title("Gripper")
  ax_gripper.set_xlim(0,1)  # initialize a bar from 0 to 1
  ax_gripper.set_yticks([]) # no y ticks
  ax_gripper.set_xticks([0,1]) # two ticks for the x axis
  ax_gripper.set_xticklabels(['Closed', 'Open'])
  bar = ax_gripper.barh(y=[0], width=[gripper_norm[0]], color='black', height=0.1) # initialization of the bar

  # define a nested "update" function to update ax_video, ax_3dplot and ax_gripper (t is the step)
  def update(t):

    # Update the ax_video
    im.set_array(frames[t])

    # Update of the ax_3dplot
    nonlocal q_x, q_y, q_z # nonlocal says to update the previous defined variables
    # Firstly remove the previous arrows
    q_x.remove()
    q_y.remove()
    q_z.remove()
    # Extract the pos (x,y,z values where to origin the vectors at frame t)
    pos = traj[t]
    # Extract the ori Axis Angle values and build the corresponding rotation Matrix
    # the columns of the Rotation Matrix are orthonormal vector that compose the U,V,W components in quiver!
    # R_Matrix = [q_x, q_y, q_z], each vector is a column! It means we take U,V,W as the components of that column
    R_Matrix = R.from_rotvec(ori[t]).as_matrix()
    q_x = ax_3dplot.quiver(X=pos[0], Y=pos[1], Z=pos[2], U=R_Matrix[0,0], V=R_Matrix[1,0], W=R_Matrix[2,0], color='r', length=0.05)  # x,y,z are the point of origin of the vector, u,v,w are the components of the vector
    q_y = ax_3dplot.quiver(X=pos[0], Y=pos[1], Z=pos[2], U=R_Matrix[0,1], V=R_Matrix[1,1], W=R_Matrix[2,1], color='g', length=0.05)
    q_z = ax_3dplot.quiver(X=pos[0], Y=pos[1], Z=pos[2], U=R_Matrix[0,2], V=R_Matrix[1,2], W=R_Matrix[2,2], color='b', length=0.05)

    # Update of the ax_gripper
    bar[0].set_width(gripper_norm[t])

    return im, q_x, q_y, q_z, bar
  
  ani = FuncAnimation(fig=fig, func=update, frames=len(frames), interval=50, blit=False)
  plt.tight_layout()
  plt.close(fig)
  return ani


'''
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


  if dataset == 'droid':
    # Take all language instructions and create a list with all the texts
    language_instructions = [steps[0][key].numpy().decode('utf-8') for key in lang_keys]
    print(language_instructions)
  elif dataset == 'bridge':
    language_instructions = [steps[0]['observation'][key].numpy().decode('utf-8') for key in lang_keys]


  
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
'''

   
   

    
   
      
      
      

    
      


    