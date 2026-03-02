import torch
import numpy as np
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel


class VJEPABackbone(nn.Module):
    def __init__(self, model_path):
        super(VJEPABackbone, self).__init__()

        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True) 
        # Load the predictor and encoder from the model
        self.predictor = self.model.predictor
        self.encoder = self.model.encoder
        # Load the processor: 
        self.processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
    
    def get_model_info(self):
        return {
            "model": self.model,
            "predictor": self.predictor,
            "encoder": self.encoder,
            "processor": self.processor
        }


    def preprocess_frames(self, frames):
        # Preprocess the input frames using the processor
        inputs = self.processor(videos=frames, return_tensors="pt")
        return inputs

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
    

if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path = "checkpoints/facebook/vjepa2-vith-fpc64-256" # the local path where the model weights are saved, for example "checkpoints/facebook/vjepa2-vith-fpc64-256"
    vjepa_backbone = VJEPABackbone(model_path)
    frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)]
    inputs = vjepa_backbone.preprocess_frames(frames)
    outputs = vjepa_backbone(inputs['pixel_values_videos'])
    