import argparse 
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils import get_libero_path
import os
import imageio 

# Argument parsing to configure rendering and video saving 
# You can run the script with "python main.py --render {True,False} --save_video {True,False}" to render/not render the simulation and to save/not save a video at the end
argparse = argparse.ArgumentParser(description="Args for video saving and rendering")
argparse.add_argument("--render", type=bool, default=True, help="True if you want to render the simulation")
argparse.add_argument("--save_video", type=bool, default=True, help="True if you want to save video of the simulation")


# Configuration to use the render mode or to save video
RENDER_MODE = argparse.parse_args().render
SAVE_VIDEO = argparse.parse_args().save_video
RENDER_CAMERA = "agentview" # the camera name used for rendering and saving video, can be "agentview", "robot0_eye_in_hand", etc. depending on the task
print(f"[info] RENDER_MODE: {RENDER_MODE}, SAVE_VIDEO: {SAVE_VIDEO}, RENDER_CAMERA: {RENDER_CAMERA}\n")


benchmark_dict = benchmark.get_benchmark_dict() # dictionary of the type {"<task_suite_name>": <task_suite>}
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]() # task suite is a set of different tasks, we'll retrieve a specific task with the task_id

# retrieve a specific task
task_id = 0 # id of the task to retrieve
task = task_suite.get_task(task_id) # task is the object with all the information about the task
task_name = task.name # the name of the task as "KITCHEN_SCENE_1_put_the_black_bowl_at_the_front_on_the_coffee_table"
task_description = task.language # instruction in natural language ("Es. put the black bowl at the front on the coffee table")
print(f"\n[info] retrieving task {task_id} from suite {task_suite_name}\n")
print(f"[info] task description: {task_description}\n")


task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 1024,
    "camera_widths": 1024,
    #"camera_names": ["agentview"],
    "has_renderer": RENDER_MODE, # If True, open the MuJoCo screen to render the env
    "has_offscreen_renderer": SAVE_VIDEO, # If True, save images rendered to create a video 
    "use_camera_obs": SAVE_VIDEO, # If True, the "obs" will include camera observations (e.g., RGB images) that can be used to create a video. 
    "render_camera": RENDER_CAMERA, # the camera name used for rendering and saving video
}

env = ControlEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
frames = [] # list used to store frames for video saving

# Loop over the environment to apply actions and collect observations
for step in range(1000):
    obs, reward, done, _ = env.step(dummy_action)
    if RENDER_MODE:
        env.env.viewer.render()
    if SAVE_VIDEO:
        frames.append(obs["agentview_image"][::-1])

# At the end the env is closed
env.close()

# Save video if required to a task.mp4 file
if SAVE_VIDEO and len(frames) > 0:
    imageio.mimsave("task.mp4", frames, fps=60)
    print("Video saved: task.mp4")