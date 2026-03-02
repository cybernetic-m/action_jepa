# Author: Massimo Romano
# Master Thesis - Sapienza University of Rome
# Title: ""
# Data: 2026

import os
import requests
from tqdm import tqdm

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


# --- GET WEIGHTS FUNCTION ---- #
def get_weights():
# Description: Function to download the weights of the V-JEPA 2 AC model from Zenodo at the link "https://zenodo.org/records/18834364"
# and save them to the local directory "checkpoints".

    # The filename of the model to be downloaded and the target directory where the file will be saved
    filename = "vjepa2_ac_droid.pth.tar"
    target_dir = "checkpoints"
    local_path = os.path.join(target_dir, filename) # the full local path where the file will be saved (Ex. checkpoints/vjepa2_ac_droid.pth.tar)

    # Create the target directory if it does not exist, otherwise print that the directory already exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    else: 
        print(f"Directory already exists: {target_dir}")
    
    # Check if the file already exists at the local path. 
    if os.path.exists(local_path):
        print(f"File already exists at: {local_path}")
        return local_path
    
    # The Zenodo URL for the file to be downloaded
    url = f"https://zenodo.org/records/18834364/files/{filename}?download=1" # direct download link for the file on Zenodo

    # Call the download function to download the file from the specified URL and save it to the local path
    download_file(url, local_path)
    return local_path

if __name__ == "__main__":
    # Call the function to download the V-JEPA 2 weights and save them to the local directory "checkpoints"
    weights_path = get_weights()
    print(f"Pretrained model weights downloaded and saved at: {weights_path}")
