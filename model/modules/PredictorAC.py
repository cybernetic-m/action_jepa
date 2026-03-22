import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)


jepa_wms_path = os.path.join(project_root, "jepa-wms")
if jepa_wms_path not in sys.path:
    sys.path.append(jepa_wms_path)

import torch
import numpy as np
import torch.nn as nn
from src.models.ac_predictor import VisionTransformerPredictorAC

class PredictorAC(nn.Module):
    def __init__(self, model_path, frozen=True, device="cpu"):
        super(PredictorAC, self).__init__()

        # Setting the device (ex. cuda or cpu)
        self.device = device

        self.frozen = frozen
        
        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        if os.path.exists(model_path):
            
            self.predictor = VisionTransformerPredictorAC(
                embed_dim=1408,
                predictor_embed_dim=1024,
                action_embed_dim=7,
                img_size=(256, 256),
                patch_size = 16,
                depth=24
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['predictor']
            
            # The state_dict keys are saved as "module.predictor_blocks.1.mlp.fc1.weight"
            # Here we remove module. part if present to have a state dict with keys as "predictor_blocks.1.mlp.fc1.weight"
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v

            self.predictor.load_state_dict(state_dict=new_state_dict)
            print(self.predictor)
        else:
            raise FileNotFoundError(f"V-JEPA AC Predictor not found in {model_path}. Run 'python download_models.py' to download it!")
        
        # Freeze the V-JEPA 2 image encoder parameters 
        if self.frozen:
            for param in self.predictor.parameters():
                param.requires_grad = False
            self.predictor.eval() # Put the encoder in evaluation model
        else:
            for param in self.predictor.parameters():
                param.requires_grad = True
            self.predictor.train() # Put the encoder in evaluation model
    

    def forward(self, x, actions, states = None, extrinsics = None):

        B, N, D = x.shape
        T = N // 256
        
        if states is None:
            states = torch.zeros(B, T, 1, 1024, device=x.device)

        x, action_features, proprio_features = self.predictor(x, actions, states, extrinsics)
        
        return x, action_features, proprio_features
    
'''
if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path= "checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar" 
    vjepa_encoder = PredictorAC(model_path=model_path, device='cuda')
'''

    