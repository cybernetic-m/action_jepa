from transformers import AutoVideoProcessor, AutoModel

model_path = "./checkpoints/facebook/vjepa2-vith-fpc64-256" # the local path where the downloaded model weights are saved, for example "checkpoints/facebook/vjepa2-vith-fpc64-256"

model = AutoModel.from_pretrained(model_path, local_files_only=True) # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)

print(model)