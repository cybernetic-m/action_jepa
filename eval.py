import sys
import os
current_dir = os.getcwd()
libero_path = os.path.join(current_dir, "LIBERO")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils import get_libero_path
import numpy as np
import imageio
import torch
from collections import deque
import random
from model.TransformerActionJEPA5 import TransformerActionJEPA
from tqdm import tqdm
import cv2
import json

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using the device: {device}")

    policy_dir_path = './results/results_alcor_6/2026_06_24__17_41'


    RENDER_CAMERA = "agentview" 

    checkpoints_path = './checkpoints'

    model_path = os.path.join(policy_dir_path, 'best_model.pth')
    config_path = os.path.join(policy_dir_path, 'config.json')
    metrics_path = os.path.join(policy_dir_path, 'metrics.csv')

    with open(config_path, 'r') as f:
        config = json.load(f)

    DATASETS = config['training']['datasets']
    POLICY_TYPE = config['model']['policy']

    NUM_FRAMES          = config['model']['num_frames']
    ACTION_CHUNK_SIZE = config['model']['action_chunk_size']
    EMBED_DIM           = config['model']['embed_dim']
    TRANSFORMER_LAYERS  = config['model']['transformer_layers']
    TRANSFORMER_HEADS   = config['model']['transformer_heads']
    TRANSFORMER_FF_DIM  = config['model']['transformer_ff_dim']
    TRANSFORMER_DROPOUT = config['model']['transformer_dropout']
    MLP_HIDDEN_DIMS     = config['model']['mlp_hidden_dims']
    MLP_DROPOUT         = config['model']['mlp_dropout']
    FROZEN_BACKBONE    = config['model']['frozen_backbone']
    AGGREGATION_MODE = config['model']['aggregation_mode']

    print(f" Camera initialized on: {RENDER_CAMERA}")
    print(f" Task suite: {DATASETS}")
    print(f"Num frames: {NUM_FRAMES}")
    print(f"Action Chunk Size: {ACTION_CHUNK_SIZE}")
    print(f"Aggregation Mode: {AGGREGATION_MODE}")

    # Path of the models V-JEPA 2 Encoder, CLIP Encoder and V-JEPA 2 AC Predictor
    vjepa_path = os.path.join(checkpoints_path,"facebook/vjepa2-vitg-fpc64-256")
    predictor_path = os.path.join(checkpoints_path,"facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar")
    clip_path = os.path.join(checkpoints_path,"openai/clip-vit-large-patch14")

    # Path of your model to test
    print("Loading model...")
    model = torch.load(model_path, map_location='cpu')
    checkpoint = model['model_state_dict']

    model = TransformerActionJEPA(
        vjepa_encoder_path=vjepa_path,
        vjepa_predictor_path=predictor_path,
        clip_model_path=clip_path,
        num_frames=NUM_FRAMES,
        action_chunk_size = ACTION_CHUNK_SIZE,
        embed_dim = EMBED_DIM,
        transformer_layers = TRANSFORMER_LAYERS,
        transformer_heads = TRANSFORMER_HEADS,
        transformer_ff_dim = TRANSFORMER_FF_DIM,
        transformer_dropout = TRANSFORMER_DROPOUT,
        mlp_hidden_dims = MLP_HIDDEN_DIMS,
        mlp_dropout = MLP_DROPOUT,
        frozen_backbone = FROZEN_BACKBONE,
        #aggregation_mode = AGGREGATION_MODE,
        device=device,
    ).to(device)

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()


    print("="*50 + "\n")
    print(f"[info] RENDER_CAMERA: {RENDER_CAMERA}\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = DATASETS[0]
    task_suite = benchmark_dict[task_suite_name]()
    n_tasks = 10
    print(f"[info] The benchmark {DATASETS[0]} has {task_suite.get_num_tasks()} tasks. Evaluating on first {n_tasks}.")

    total_episodes_per_task = 50
    MAX_VIDEOS_PER_TASK = 4

    # Variabili per tracciare il successo globale separatamente
    global_success = {"actor": 0, "refiner": 0}

    # Due dizionari separati per i report JSON
    metrics_report_actor = {
        "benchmark_name": task_suite_name,
        "model_tested": "ACTOR_BASELINE",
        "total_demos_planned": n_tasks * total_episodes_per_task,
        "global_success_count": 0,
        "overall_success_rate": 0.0,
        "tasks_details": {}
    }

    metrics_report_refiner = {
        "benchmark_name": task_suite_name,
        "model_tested": "REFINER",
        "total_demos_planned": n_tasks * total_episodes_per_task,
        "global_success_count": 0,
        "overall_success_rate": 0.0,
        "tasks_details": {}
    }

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        print(f"\n{'='*50}")
        print(f"[info] Evaluating Task {task_id}: {task_description}")
        print(f"{'='*50}")

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": ["agentview"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "render_camera": RENDER_CAMERA,
            "controller": "OSC_POSE",
            "control_freq": 20
        }

        env = ControlEnv(**env_args)
        env.seed(0)
        init_states = task_suite.get_task_init_states(task_id)

        # Crea le sottocartelle dedicate a questo task per i video
        task_folder_name = f"{task_suite_name}_task{task_id}_{task_description.replace(' ', '_')}"
        task_video_dir_actor = os.path.join(policy_dir_path, task_folder_name, "actor_videos")
        task_video_dir_refiner = os.path.join(policy_dir_path, task_folder_name, "refiner_videos")
        os.makedirs(task_video_dir_actor, exist_ok=True)
        os.makedirs(task_video_dir_refiner, exist_ok=True)

        task_success_count = {"actor": 0, "refiner": 0}
        saved_videos_count = {"actor": 0, "refiner": 0}

        for ep in range(total_episodes_per_task):
            
            
            for policy_type in ["actor", "refiner"]:
                
                env.reset()
                
                env.set_init_state(init_states[ep])

                window_size = model.num_frames
                frame_buffer = deque(maxlen=window_size)
                text_input = task_description
                video_frames = []

                init_action = [0.] * 7
                obs, _, _, _ = env.step(init_action)
                
                # Riempiamo il buffer visivo iniziale
                for _ in range(window_size):
                    obs, _, _, _ = env.step(init_action)
                    env_frame = np.flip(obs['agentview_image'], axis=0).copy()
                    frame_buffer.append(env_frame)

                model.use_backbone = True
                
                with torch.no_grad():
                    max_steps = 250
                    step = 0
                    is_success = False
                    
                    pbar = tqdm(total=max_steps, desc=f"Ep {ep+1}/{total_episodes_per_task} [{policy_type.upper()}]", leave=False)
                    
                    while step < max_steps:
                        
                        joint_input = torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).float().to(device)
                        vision_tensor = np.stack(list(frame_buffer), axis=0)
                        vision_input = torch.from_numpy(vision_tensor).byte().unsqueeze(0).to(device)
                        
                        actor_action, refiner_action, _ = model(text_input, vision_input, joint_input)
                        
                        if policy_type == "actor":
                            chunk_actions = actor_action[0].cpu().numpy()
                        else:
                            chunk_actions = refiner_action[0].cpu().numpy()
                        
                        chunk_size = chunk_actions.shape[0]
                        
                        for j in range(chunk_size):
                            if step >= max_steps:
                                break
                
                            vla_action = chunk_actions[j]
                            next_obs, reward, done, info = env.step(vla_action)

                            next_frame = np.flip(next_obs['agentview_image'], axis=0).copy()
                            video_frames.append(next_frame)
                            frame_buffer.append(next_frame)
                                
                            obs = next_obs
                            step += 1
                            pbar.update(1)

                            if done or info.get("success", False):
                                is_success = True
                                break
                        
                        if is_success:
                            break

                    pbar.close()

                    if is_success:
                        task_success_count[policy_type] += 1
                        global_success[policy_type] += 1
                        print(f"✅ Ep {ep+1:02d} [{policy_type.upper()}] - Success!")

                        if saved_videos_count[policy_type] < MAX_VIDEOS_PER_TASK and len(video_frames) > 0:
                            videoname = f"{policy_type}_success_ep{ep+1:02d}.mp4"
                            target_dir = task_video_dir_actor if policy_type == "actor" else task_video_dir_refiner
                            imageio.mimsave(os.path.join(target_dir, videoname), video_frames, fps=60)
                            saved_videos_count[policy_type] += 1
                    else:
                        print(f"❌ Ep {ep+1:02d} [{policy_type.upper()}] - Failed.")

        env.close()
        
        sr_actor = (task_success_count["actor"] / total_episodes_per_task) * 100
        sr_refiner = (task_success_count["refiner"] / total_episodes_per_task) * 100
        
        print(f"\n📊 [Result Task {task_id}] Actor SR: {task_success_count['actor']}/{total_episodes_per_task} ({sr_actor:.1f}%)")
        print(f"📊 [Result Task {task_id}] Refiner SR: {task_success_count['refiner']}/{total_episodes_per_task} ({sr_refiner:.1f}%)")

        metrics_report_actor["tasks_details"][task_folder_name] = {
            "task_id": task_id,
            "description": task_description,
            "success_count": task_success_count["actor"],
            "total_episodes": total_episodes_per_task,
            "success_rate_percentage": float(sr_actor)
        }
        
        metrics_report_refiner["tasks_details"][task_folder_name] = {
            "task_id": task_id,
            "description": task_description,
            "success_count": task_success_count["refiner"],
            "total_episodes": total_episodes_per_task,
            "success_rate_percentage": float(sr_refiner)
        }


    total_evaluations = n_tasks * total_episodes_per_task

    overall_sr_actor = (global_success["actor"] / total_evaluations) * 100
    metrics_report_actor["global_success_count"] = global_success["actor"]
    metrics_report_actor["total_demos_executed"] = total_evaluations
    metrics_report_actor["overall_success_rate"] = float(overall_sr_actor)

    overall_sr_refiner = (global_success["refiner"] / total_evaluations) * 100
    metrics_report_refiner["global_success_count"] = global_success["refiner"]
    metrics_report_refiner["total_demos_executed"] = total_evaluations
    metrics_report_refiner["overall_success_rate"] = float(overall_sr_refiner)

    print("\n" + "="*50)
    print(f"🏆 OVERALL ACTOR SUCCESS RATE:   {global_success['actor']}/{total_evaluations} ({overall_sr_actor:.2f}%)")
    print(f"🏆 OVERALL REFINER SUCCESS RATE: {global_success['refiner']}/{total_evaluations} ({overall_sr_refiner:.2f}%)")
    print("="*50 + "\n")

    json_actor_path = os.path.join(policy_dir_path, f"{task_suite_name}_actor_metrics.json")
    with open(json_actor_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report_actor, f, indent=4, ensure_ascii=False)

    json_refiner_path = os.path.join(policy_dir_path, f"{task_suite_name}_refiner_metrics.json")
    with open(json_refiner_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report_refiner, f, indent=4, ensure_ascii=False)

    print(f"📄 JSON metrics successfully saved to:\n - {json_actor_path}\n - {json_refiner_path}")


