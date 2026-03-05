import torch
import numpy as np
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel


class VJEPAEncoder(nn.Module):
    def __init__(self, model_path):
        super(VJEPAEncoder, self).__init__()

        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True) 
        # Load the encoder from the model
        self.encoder = self.model.encoder
        # Load the processor
        self.processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
    
    def get_model_info(self):
        return {
            "encoder": self.encoder,
            "processor": self.processor
        }


    def preprocess_frames(self, frames):
        # Preprocess the input frames using the processor
        inputs = self.processor(videos=frames, return_tensors="pt")
        return inputs

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs
    

if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path = "checkpoints/facebook/vjepa2-vith-fpc64-256" # the local path where the model weights are saved, for example "checkpoints/facebook/vjepa2-vith-fpc64-256"
    vjepa_encoder = VJEPAEncoder(model_path)
    frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)]
    inputs = vjepa_encoder.preprocess_frames(frames)
    outputs = vjepa_encoder(inputs['pixel_values_videos'])
    print(type(outputs))
    