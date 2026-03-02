# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import os
import requests
from tqdm import tqdm
from transformers import AutoVideoProcessor, AutoModel
from huggingface_hub import snapshot_download

# --- DOWNLOAD FUNCTION ---- #
def download_file(url, output_path):
# Description: Download a file from a given URL and save it to the specified output path.
# Args:
#   url (str): The URL of the file to be downloaded.
#   output_path (str): The local file path where the downloaded file will be saved.
    try:
        # Send a GET request to the specified URL with streaming enabled (it allows downloading large files without loading them entirely into RAM memory)
        response = requests.get(url, stream=True)

        # Get the total size of the file from the response headers, this is used for tqdm bar
        total_size = int(response.headers.get('content-length', 0)) 

        # Open the output file in binary write mode and write the content of the response to it in chunks, while updating the tqdm progress bar
        # The tqdm bar will divide the total size of the file in 1024, 1024*1024, etc. and will display the most scaled unit (KB, MB, etc.) in the progress bar
        # Es. File size = 100 000 000 bytes, tqdm will display the progress bar in MB (100 000 000 / 1024 / 1024 = 95.37 MB) 
        with open(output_path, 'wb') as file:
            with tqdm(desc=f"Downloading {output_path}", total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data) # write the chunk of data to the file, returning the size of the chunk just written
                    bar.update(size) # update the tqdm bar with the size of the chunk just written
        print(f"File downloaded successfully and saved to: {output_path}")

    # Handling different types of exceptions that may occur during the download process and printing appropriate error messages
    # For example 404 Not Found, connection issues, timeouts, or generic errors.
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading the file: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred while downloading the file: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred while downloading the file: {timeout_err}")
    except Exception as err:
        print(f"A generic error occurred while downloading the file: {err}")


# --- DOWNLOAD MODELS FUNCTION ---- #
def hf_download_models(hf_repo_id_list):
# Description: Function to download the weights of different models 
# and save them to the local directory "checkpoints".
# Args:
#   model_list (list of str): A list of model names to be downloaded.
#   model_urls (list of str): A list of URLs corresponding to the models in model_list

    target_dir = "checkpoints" # the directory where the downloaded models will be saved 

    for repo_id in hf_repo_id_list:
        print(f"Downloading model from Hugging Face repository: {repo_id}")
        
        local_dir = os.path.join(target_dir, repo_id) # the full local path where the file will be saved (Ex. checkpoints/vjepa2_ac_droid.pth.tar)

        # Create the target directory if it does not exist, otherwise print that the directory already exists
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created directory: {target_dir}")
        else: 
            print(f"Directory already exists: {target_dir}")
            continue
        
        # Check if the file already exists at the local path. 
        if os.path.exists(local_dir):
            print(f"File already exists at: {local_dir}")
            continue

        # Call the download function of Hugging Face to download the file from the specified URL and save it to the local path
        snapshot_download(
            repo_id=repo_id, # the identifier of the Hugging Face repository to download from, for example "facebook/vjepa2-vitg-fpc64-256"
            local_dir=local_dir, # the local directory where the downloaded files will be saved, in this case "checkpoints"
            local_dir_use_symlinks=False, # if True, the downloaded files will be saved as symbolic links to the cache directory, if False the files will be copied to the local directory.
            revision="main", # the branch, tag or git identifier to download from the repository, in this case we want to download from the main branch
        )
    return target_dir


if __name__ == "__main__":
    # Call the function to download the V-JEPA 2 weights and save them to the local directory "checkpoints"
    weights_path = hf_download_models(
                    hf_repo_id_list=[
                        "facebook/vjepa2-vith-fpc64-256",
                        "google-bert/bert-base-uncased"
                    )
    print(f"Pretrained model weights downloaded and saved at: {weights_path}")
