# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import os
from huggingface_hub import snapshot_download, hf_hub_download

# --- DOWNLOAD MODELS FUNCTION ---- #
def hf_download_models(hf_model_list):
# Description: Function to download the weights of different models 
# and save them to the local directory "checkpoints".
# Args:
#   hf_model_list: A list of Hugging Face repository identifiers to download the models from, for example ["facebook/vjepa2-vith-fpc64-256", "google-bert/bert-base-uncased"]
# Returns:
#   target_dir: The local directory where the downloaded models are saved, in this case "checkpoints"

    target_dir = "checkpoints" # the directory where the downloaded models will be saved 

    # Create the target directory if it does not exist, otherwise print that the directory already exists
    os.makedirs(target_dir, exist_ok=True)
    
    for item in hf_model_list:
        print(f"\nDownloading model from Hugging Face repository: {item}\n")
        
        local_dir = os.path.join(target_dir, item) # the full local path where the file will be saved (Ex. checkpoints/vjepa2_ac_droid.pth.tar)

        # Check if the file already exists at the local path. 
        if os.path.exists(local_dir):
            print(f"\nFile already exists at: {local_dir}\n")
            continue
        
        if ".pth.tar" in item:
            parts = item.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])

            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir
            )

        else:
            # Call the download function of Hugging Face to download the file from the specified URL and save it to the local path
            snapshot_download(
                repo_id=item, # the identifier of the Hugging Face repository to download from, for example "facebook/vjepa2-vitg-fpc64-256"
                local_dir=local_dir, # the local directory where the downloaded files will be saved, in this case "checkpoints"
                local_dir_use_symlinks=False, # if True, the downloaded files will be saved as symbolic links to the cache directory, if False the files will be copied to the local directory.
                revision="main", # the branch, tag or git identifier to download from the repository, in this case we want to download from the main branch
            )
    return target_dir


if __name__ == "__main__":
    # Call the function to download the V-JEPA 2 weights and save them to the local directory "checkpoints"
    weights_path = hf_download_models(
                    hf_model_list=[
                        "facebook/vjepa2-vith-fpc64-256",
                        "openai/clip-vit-large-patch14",
                        "facebook/jepa-wms/vjepa2_ac_droid.pth.tar"
                      
                    ]
                    )
    print(f"\nPretrained model weights downloaded and saved at: {weights_path}\n")

    import torch

    checkpoint_path = "./checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar" 
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

