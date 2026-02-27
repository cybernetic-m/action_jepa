import sys
import os
# Define the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the LIBERO directory (assuming it is in the same directory as this main file)
libero_path = os.path.join(current_dir, "LIBERO")
# Add the LIBERO directory to the system path if it's not already there, so that we can import LIBERO modules
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)


import argparse
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils import get_libero_path
import imageio 
import numpy as np


# Argument parsing to configure rendering
# You can run the script with "python main.py --render" to render the simulation.
# If you do not want to render the simulation simply omit the flag.
argparse = argparse.ArgumentParser(description="Args for video saving and rendering")
argparse.add_argument("--render", action="store_true", help="If you want to render the simulation write the flag --render")

# Configuration to use the render mode
RENDER_MODE = argparse.parse_args().render
RENDER_CAMERA = "agentview" # the camera name used for rendering, can be "agentview", "robot0_eye_in_hand", etc. depending on the task
print(f"[info] RENDER_MODE: {RENDER_MODE}, RENDER_CAMERA: {RENDER_CAMERA}\n")


benchmark_dict = benchmark.get_benchmark_dict() # dictionary of the type {"<task_suite_name>": <task_suite_class>} (Ex. "libero_spatial": <class 'libero.libero.benchmark.LIBERO_SPATIAL'>)
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]() # task suite is a class that represents a set of different tasks, we'll retrieve a specific task with the task_id

# retrieve a specific task
task_id = 0 # id of the task to retrieve
task = task_suite.get_task(task_id) # task is the object with all the information about the specific task
task_name = task.name # the name of the task as "KITCHEN_SCENE_1_put_the_black_bowl_at_the_front_on_the_coffee_table"
task_description = task.language # instruction in natural language ("Es. put the black bowl at the front on the coffee table")
print(f"\n[info] retrieving task {task_id} from suite {task_suite_name}\n")
print(f"[info] task description: {task_description}\n")

# retrieve the BDDL (Behavioral Design Definition Language) file for the task, it is a file containing all the informations about objects and initial and goal state 
# (Ex. On Bowl1 Table1 means that the bowl 1 is on the table at initial state for example)
# get_libero_path("bddl_files"): find the directory where BDDL files are stored in your pc (Ex. /home/user/Desktop/LIBERO/libero/libero/./bddl_files )
# task.problem_folder: the folder containing the BDDL file for the specific task (Ex. libero_spatial)
# task.bddl_file: the name of the BDDL file for the specific task (Ex. pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] BDDL file for the task: {task_bddl_file}\n")

# Define the text instruction for the VLA
text_instruction = f"In: What action should the robot take to {task_description}?\nOut:"

# Args for environment initialization
env_args = {
    "bddl_file_name": task_bddl_file, # path of the BDDL file
    "camera_heights": 256, 
    "camera_widths": 256,
    "camera_names": ["agentview"],
    "has_renderer": RENDER_MODE, # If True, open the MuJoCo screen to render the env
    "has_offscreen_renderer": True, # If True, save images rendered to create a video 
    "use_camera_obs": True, # If True, the "obs" will include camera observations (e.g., RGB images) that can be used to create a video. 
    "render_camera": RENDER_CAMERA, # the camera name used for rendering and saving video
}

env = ControlEnv(**env_args) # create the env class
env.seed(0) # set a seed for reproducibility
env.reset() # reset the scene and bring to initial state

# init_states is a list of 50 possible different intial states
# each init_state is a vector of state formed by joint positions, objects positions and orientations, velocities ...
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 10 # Among all 50 initial states you can spawn objects in different ways choosing init_state_id (it can be a int number in [0,49])
env.set_init_state(init_states[init_state_id]) # set the init_state chosen

init_action = [0.] * 7 # a start action needed for the first step, to have the first observation from the environment
frames = [] # list used to store frames for video saving

# Loop over the environment to apply actions and collect observations
# Obs is an ordered dict that have information about the ROBOT:
# - "agentview_image": the RGB image from the agent's camera => camera_heights x camera_widths x 3 (ex 1024 x 1024 x 3)
# - "robot0_joint_positions": array of vlues of the robot's joint positions 
# - "robot0_joint_pos_cos (or sin)": array of cos (or sin) values of the robot's joint positions
# - "robot0_joint_vel": array of values of the robot's joint velocities
# - "robot0_eef_pos": array of values of the robot's end effector position (x,y,z)
# - "robot0_eef_quat": array of values of the robot's end effector orientation in quaternion (x,y,z,w)
# - "robot0_gripper_qpos": the gripper's position (two values for the two fingers distances from the center, 0 means fully closed)
# - "robot0_gripper_qvel": the gripper's velocity (two values for the two fingers velocities)
# And information about the object. Each object called "objectName" has two arrays:
# - "objectName_pos": array of values of the object's position (x,y,z)
# - "objectName_quat": array of values of the object's orientation in quaternion (x,y,z,w)
# - "objectName_to_robot0_eef_pos": array of values of the object's position relative to the robot's end effector (x,y,z)
# - "objectName_to_robot0_eef_quat": array of values of the object's orientation relative to the robot's end effector (x,y,z,w)

obs, _, _, _ = env.step(init_action) # apply the first zero action to have the first observation from the environment

for step in range(100):
    print(f"\nStep {step}\n")
    image = obs["agentview_image"] # get the RGB image from the agent's camera

    # At the moment a zero action 
    action = [0.0] * 7
    print(f"Action predicted by the VLA: {action}\n")

    obs, reward, done, _ = env.step(action)
    if RENDER_MODE:
        env.env.viewer.render()
    frames.append(np.flipud(obs["agentview_image"])) # np.flipud means Flip Up Down, it is used to flip the image vertically before appending

# At the end the env is closed
env.close()

# Save video if required to a task.mp4 file
if len(frames) > 0:
    imageio.mimsave("task.mp4", frames, fps=60)
    print("Video saved: task.mp4")