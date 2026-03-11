import torch
import torch.nn as nn
import os
import numpy as np

from modules import CLIPBackbone, VJEPABackbone, Predictor

class SemanticJEPA(nn.Module):
    def __init__(self, 
                 vjepa_model_path, 
                 clip_model_path, 
                 vision_tokens = 2048, 
                 language_tokens = 77, 
                 state_tokens = 1, 
                 action_tokens = 1,
                 vision_dim = 1280, 
                 language_dim = 768 , 
                 action_dim = 7, 
                 state_dim=7, 
                 predictor_dim = 1024, 
                 num_layers = 12,
                 device = 'cpu',
                 ):
        super(SemanticJEPA, self).__init__()

        # Define the V-JEPA 2 Vision Backbone and the CLIP Language Backbone, both are frozen during training
        self.vision_backbone = VJEPABackbone(model_path = vjepa_model_path, device = device)
        self.language_backbone = CLIPBackbone(model_path = clip_model_path, device = device)

        # Define the Predictor that is the module that we train
        self.predictor = Predictor(
                 vision_tokens = vision_tokens, 
                 language_tokens = language_tokens, 
                 state_tokens = state_tokens, 
                 action_tokens = action_tokens,
                 vision_dim = vision_dim, 
                 language_dim = language_dim, 
                 action_dim = action_dim, 
                 state_dim=state_dim, 
                 predictor_dim = predictor_dim, 
                 num_layers = num_layers,
        ).to(device)

    def forward(self, vision_input, text_input, state, action):

        # Firstly we preprocess frames of the video and pass the result to the V-JEPA 2 Vision Backbone
        # In particular we will pass during training directly the embeddings [B, seq_len, vision_dim] 
        # (we have saved "offline" in a file the values after passing a video to "preprocess_frames" and the result to V JEPA 2 Backbone, the frozen vision part) 
        # Training (we check if we have a tensor of dimension 3 => [B, seq_len, vision_dim])
        if torch.is_tensor(vision_input) and vision_input.dim() ==3:
            z_obs = vision_input
        # Inference (we pass directly the video that we preprocess and embed through V-JEPA 2 Backbone in real time)
        else:
            pixel_video_values = self.vision_backbone.preprocess_frames(vision_input)
            z_obs = self.vision_backbone(pixel_video_values)

        # Secondly we use CLIP to produce embeddings of the text_instruction (same reasoning fro training and inference of before)
        # Training
        if torch.is_tensor(text_input) and text_input.dim() ==3:
            z_goal = text_input
        # Inference 
        else:
            text_tokens = self.language_backbone.tokenization(text_input)
            z_goal = self.language_backbone(text_tokens)

        return self.predictor(z_goal, state, action, z_obs)
    
if __name__== "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vjepa_path = "checkpoints/facebook/vjepa2-vith-fpc64-256"
    clip_path = "checkpoints/openai/clip-vit-large-patch14"
    semantic_jepa = SemanticJEPA(
        vjepa_model_path=vjepa_path,
        clip_model_path=clip_path,
        device=device
    )

    state = torch.randn(1, 7).to(device)
    action = torch.randn(1, 7).to(device)

    
    # TEST OF THE TRAINING PHASE
    z_obs_train = torch.randn(1, 2048, 1280).to(device)
    z_goal_train = torch.randn(1, 77, 768).to(device)

    z_next_pred_train = semantic_jepa(z_obs_train, z_goal_train, state, action)

    # TEST OF THE INFERENCE PHASE
    
    raw_frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)]
    raw_text = "Pick up the red cube and place it to the right."

    z_next_pred_inf = semantic_jepa(raw_frames, raw_text, state, action)
    