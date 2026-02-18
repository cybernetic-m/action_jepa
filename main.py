import argparse 
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils import get_libero_path
import os
import imageio 

argparse = argparse.ArgumentParser(description="Args for video saving and rendering")
argparse.addargument("--render", type=bool, default=True, help="Whether to render the")
# Configuration to use the render mode or to save video
RENDER_MODE = True 
SAVE_VIDEO = True 
RENDER_CAMERA = "agentview" # the camera name used for rendering and saving video, can be "agentview", "robot0_eye_in_hand", etc. depending on the task


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"\n[info] retrieving task {task_id} from suite {task_suite_name}\n")
print(f"[info] task description: {task_description}\n")
      

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

print(f"[info] RENDER_MODE: {RENDER_MODE}, SAVE_VIDEO: {SAVE_VIDEO}, RENDER_CAMERA: {RENDER_CAMERA}\n")
env = ControlEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
frames = [] 
for step in range(1000):
    obs, reward, done, info = env.step(dummy_action)
    if RENDER_MODE:
        env.env.viewer.render()
    if SAVE_VIDEO:
        frames.append(obs["agentview_image"][::-1])

env.close()

if SAVE_VIDEO and len(frames) > 0:
    imageio.mimsave("task.mp4", frames, fps=60)
    print("Video saved: task.mp4")