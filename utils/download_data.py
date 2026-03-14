from tqdm import tqdm
import os
import torch
import tensorflow_datasets as tfds
from utils import process_data


def download_data(dataset_list, lang_keys):
    
    for dataset in dataset_list:
        download_dir = f"./data/{dataset}" # directory where to download the pytorch tensors needed for offline training
        os.makedirs(download_dir, exist_ok=True) # Create the directory /data/name_of_dataset if not exist

        # Load in streaming the dataset from Google Storage (gs)
        # shuffle_files is False beacuse it permits to avoid opening in parallel different threads that are difficult to close
        ds = tfds.load(dataset, 
                    data_dir="gs://gresearch/robotics", 
                    split="train", 
                    shuffle_files=False, # it permits to open each file sequentially without shuffling (if True it will open in parallel different file to shuffle)
                    )


        num_episodes_to_save = 5    # num_episodes we want to save
        saved_count = 0             # count the num of episodes saved

        # Take in streaming only a subset of elements (avoiding downloading in background useless episodes)
        ds_subset = ds.take(20)
        it = iter(ds_subset) # return the dataset iterator

        pbar = tqdm(total=num_episodes_to_save, desc=f"Download {dataset_list[0]} episodes", unit="ep")

        while saved_count<num_episodes_to_save:
            try:
                episode = next(it) # return the next episode 
                #tqdm.write(f"--- Episode received from Cloud")
                video_pt, states_pt, action_pt, language_instructions = process_data(episode=episode, 
                                                                                    num_frames=16, 
                                                                                    fps=15, 
                                                                                    window_second_size=4, 
                                                                                    lang_keys=droid_lang_keys)
                if video_pt is not None:
                    sample = {
                        "video": video_pt,      # [16, 3, 256, 256]
                        "states": states_pt,    # [16, 7]
                        "actions": action_pt,   # [15, 7]
                        "language": language_instructions
                    }
                    
                    file_path = os.path.join(download_dir, f"episode_{saved_count}.pt")
                    torch.save(sample, file_path)
                    saved_count += 1
                    pbar.update(1)

            except StopIteration:
                # When the episodes from the iterator are finished, it raise a StopIteration
                print("\nStopIteration")
                break
            except Exception as e:
                print(f"Error during streaming: {e}")


        pbar.close()
        print(f"\n Download finished: {saved_count} episodes saved from {dataset_list[0]} dataset")


if __name__ == "__main__":
    # List of the dataset of OXE to download
    dataset_list = ["droid"]
    # The list of language instruction keys for each dataset
    droid_lang_keys = ['language_instruction', 'language_instruction_2', 'language_instruction_3']