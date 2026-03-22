import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout = 0):
        super(MLP, self).__init__()
        
        # Define a list of layers to append each layer
        layers = []

        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())

            if dropout>0:
                layers.append(nn.Dropout(dropout))

            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.mlp(x)

'''   
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    JEPA_DIM = 1408   # V-JEPA 2 ViT-G
    CLIP_DIM = 768    # CLIP ViT-L
    ACTION_DIM = 7    # 6 DoF + Gripper
    
    lang_projector = MLP(input_dim=CLIP_DIM, hidden_dims=[1024], output_dim=JEPA_DIM).to(device)
    z_clip_sample = torch.randn(1, CLIP_DIM).to(device)
    z_text_latent = lang_projector(z_clip_sample)
    print(f"lang: {lang_projector}")
  
    input_dim_p1 = JEPA_DIM + JEPA_DIM
    policy1 = MLP(input_dim=input_dim_p1, hidden_dims=[512, 256], output_dim=ACTION_DIM).to(device)
    print(f"1: {policy1}")
    
    z_obs_sample = torch.randn(1, JEPA_DIM).to(device)
    input_p1 = torch.cat([z_obs_sample, z_text_latent], dim=-1)
    a1 = policy1(input_p1)

    input_dim_p2 = JEPA_DIM + JEPA_DIM + JEPA_DIM
    policy2 = MLP(input_dim=input_dim_p2, hidden_dims=[512, 256], output_dim=ACTION_DIM).to(device)
    print(f"2: {policy2}")
    
    z_pred_sample = torch.randn(1, JEPA_DIM).to(device) 
    input_p2 = torch.cat([z_obs_sample, z_text_latent, z_pred_sample], dim=-1)
    a2 = policy2(input_p2)
'''

