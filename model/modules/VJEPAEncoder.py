import torch
import os
import numpy as np
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel


class VJEPAEncoder(nn.Module):
    def __init__(self, model_path, frozen=True, device="cpu"):
        super(VJEPAEncoder, self).__init__()

        # Setting the device (ex. cuda or cpu)
        self.device = device
        self.frozen = frozen
        
        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        if os.path.exists(model_path):
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device) # load the entire V-JEPA Model
            self.vision_encoder = self.model.encoder # Take only the V-JEPA Encoder part
            self.vision_processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True) # Load the processor
        else:
            raise FileNotFoundError(f"V-JEPA model not found in {model_path}. Run 'python download_models.py' to download it!")
        
        # Freeze the V-JEPA 2 image encoder parameters 
        if frozen:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval() # Put the encoder in evaluation mode
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            self.vision_encoder.train() # Put the encoder in train mode
            
    
    def preprocess_frames(self, video_frames):
        # Preprocess the input frames using the processor
        vision_patches = self.vision_processor(videos=video_frames, return_tensors="pt").to(self.device)
        return vision_patches['pixel_values_videos']

    def forward(self, pixel_values):
        if self.frozen:
            with torch.no_grad():
                outputs = self.vision_encoder(pixel_values)
        else: 
            outputs = self.vision_encoder(pixel_values)
        return outputs.last_hidden_state
    
'''
if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path= "checkpoints/facebook/vjepa2-vitg-fpc64-256" 
    vjepa_encoder = VJEPAEncoder(model_path=model_path, device='cuda')
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
    inputs = vjepa_encoder.preprocess_frames(frames)
    outputs = vjepa_encoder(inputs)
    print(outputs.shape)
'''