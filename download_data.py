import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Hide the GPU to TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce inutil logs


from tqdm import tqdm
import torch
import tensorflow_datasets as tfds
from utils.utils import process_data, dataset2path
import tensorflow as tf
import gc
import sys
import argparse


def download_data(dataset_list, num_episodes_to_download, lang_keys_map, fps_list, window_seconds_size_list):
    
    print("Starting downloading data...")
    
    for dataset, fps, window_seconds_size in zip(dataset_list, fps_list, window_seconds_size_list):
        
        download_dir = f"./data/{dataset}" # directory where to download the pytorch tensors needed for offline training
        os.makedirs(download_dir, exist_ok=True) # Create the directory /data/name_of_dataset if not exist
        lang_keys = lang_keys_map[dataset]
    
        # Load in streaming the dataset from Google Storage (gs)
        # shuffle_files is False beacuse it permits to avoid opening in parallel different threads that are difficult to close
        b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        ds = b.as_dataset(split='train')

        num_episodes_to_save = num_episodes_to_download   # num_episodes we want to save

        # Take in streaming only a subset of elements (avoiding downloading in background useless episodes)
        it = iter(ds) # return the dataset iterator

        # List of files existed in data
        existing_files = [f for f in os.listdir(download_dir) if f.endswith('.pt')]
        # number of files previously saved
        saved_count = len(existing_files)
        print(f"{saved_count} existing episodes in {dataset} dataset")

        if saved_count > 0:
            for _ in tqdm(range(saved_count), desc=f"Skipping existing {dataset} episodes", unit="ep"):
                try:
                    next(it)
                except StopIteration:
                    break
        
        pbar = tqdm(total=num_episodes_to_save, initial=saved_count, desc=f"Download {dataset} episodes", unit="ep", file=sys.stdout)

        while saved_count<num_episodes_to_save:
            try:
                episode = next(it) # return the next episode 
                #tqdm.write(f"--- Episode received from Cloud")
                video_pt, states_pt, action_pt, language_instructions, found_instrunction, video_extracted_seconds = process_data(episode=episode, 
                                                                                                                                  num_frames=16, 
                                                                                                                                  fps=fps, 
                                                                                                                                  window_second_size=window_seconds_size, 
                                                                                                                                  lang_keys=lang_keys,
                                                                                                                                  dataset=dataset)
                if video_pt is not None:
                    sample = {
                        "video": video_pt,      # [16, 3, 256, 256]
                        "states": states_pt,    # [16, 7]
                        "actions": action_pt,   # [15, 7]
                        "language": language_instructions,
                        "language_instructions_found": found_instrunction,
                        "video_seconds": video_extracted_seconds
                    }
                    
                    file_path = os.path.join(download_dir, f"episode_{saved_count}.pt")
                    torch.save(sample, file_path)
                    saved_count += 1    
                    pbar.update(1)

               
                del episode
                if video_pt is not None:
                    del video_pt, states_pt, action_pt
                tf.keras.backend.clear_session()
                gc.collect() 

            except StopIteration:
                # When the episodes from the iterator are finished, it raise a StopIteration
                print("\nStopIteration")
                break
            except Exception as e:
                print(f"Error during streaming: {e}")


        pbar.close()
        del ds, it
        gc.collect()
        print(f"\n Download finished: {saved_count} episodes saved from {dataset} dataset")
       
        

if __name__ == "__main__":

    argparse = argparse.ArgumentParser(description="Args for num episodes to download")
    argparse.add_argument("--num_episodes", type=int, default=1000, help="Specify the number of episodes to download for each dataset")
    
    args = argparse.parse_args()

    # List of the dataset of OXE to download
    dataset_list = ["droid", "bridge"]
    fps_list = [15,5]
    window_seconds_size_list = [4, 4]
    # The list of language instruction keys for each dataset
    lang_keys_map= {"bridge": ['natural_language_instruction'],
                    "droid": ['language_instruction', 'language_instruction_2', 'language_instruction_3'],  
    }
    

    download_data(dataset_list= dataset_list,
                  num_episodes_to_download=args.num_episodes,
                  lang_keys_map=lang_keys_map,
                  fps_list=fps_list,
                  window_seconds_size_list = window_seconds_size_list
                  )

   

   



