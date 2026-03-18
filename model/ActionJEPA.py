import torch
import torch.nn as nn
from transformers import AutoModel
from model.modules import PredictorAC
from modules import VJEPAEncoder

class ActionJEPA(nn.Module):
    def __init__(self, vjepa_encoder_path, vjepa_ac_predictor_path, device="cuda"):

        self.device = device
        
        self.vision_backbone = VJEPABackbone(model_path = vjepa_model_path, device = device)