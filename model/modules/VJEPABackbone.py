import torch
import os
import numpy as np
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel


class VJEPABackbone(nn.Module):
    def __init__(self, model_path, encoder_only = True, device="cpu"):
        super(VJEPABackbone, self).__init__()

        # Setting the device (ex. cuda or cpu)
        self.device = device
        
        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        if os.path.exists(model_path):
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device) # load the entire V-JEPA Model
            self.vision_encoder = self.model.encoder # Take only the V-JEPA Encoder part
            self.vision_processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True) # Load the processor
            # Load also the V-JEPA 2 predictor if needed ("basic" World Model mode)
            if not encoder_only:
                self.vision_predictor = self.model.predictor
        else:
            raise FileNotFoundError(f"V-JEPA model not found in {model_path}. Run 'python download_models.py' to download it!")
        
        # Freeze the V-JEPA 2 image encoder parameters 
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

            if not encoder_only:
                for param in self.vision_predictor.parameters():
                    param.requires_grad = False
        self.vision_encoder.eval() # Put the encoder in evaluation mode
    
    def preprocess_frames(self, frames):
        # Preprocess the input frames using the processor
        inputs = self.vision_processor(videos=frames, return_tensors="pt").to(self.device)
        return inputs['pixel_values_videos']

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values)
        return outputs.last_hidden_state
    

if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path_fullModel = "checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar"
    model_path_encoderOnly = "checkpoints/facebook/vjepa2-vith-fpc64-256" 
    vjepa_encoder = VJEPABackbone(device='cuda')
    frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)]
    inputs = vjepa_encoder.preprocess_frames(frames)
    outputs = vjepa_encoder(inputs)
    print(outputs.shape)
    