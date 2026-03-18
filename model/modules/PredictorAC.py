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
        
        # load the model weights from the local path, setting local_files_only to True to avoid trying to download the weights from Hugging Face if they are not found at the local path
        if os.path.exists(model_path):
            
            self.predictor = VisionTransformerPredictorAC(
                embed_dim=1408,
                predictor_embed_dim=1024,
                action_embed_dim=7,
                depth=24
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            print(checkpoint.keys())
            state_dict = checkpoint['predictor']

            # 2. FIX KEYS: Rimuoviamo il prefisso 'module.' se presente
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # rimuove 'module.' (7 caratteri)
                new_state_dict[name] = v

            self.predictor.load_state_dict(state_dict=new_state_dict)
            print(self.predictor)
        else:
            raise FileNotFoundError(f"V-JEPA AC Predictor not found in {model_path}. Run 'python download_models.py' to download it!")
        
        # Freeze the V-JEPA 2 image encoder parameters 
        if frozen:
            for param in self.predictor.parameters():
                param.requires_grad = False
            self.predictor.eval() # Put the encoder in evaluation mode
    

    def forward(self, x):
        
        return 0
    

if __name__ == "__main__":
    # Example usage of the VJEPABackbone class
    model_path= "checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar" 
    vjepa_encoder = PredictorAC(model_path=model_path, device='cuda')
    import torch

    #vjepa2_encoder, vjepa2_ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
    
    