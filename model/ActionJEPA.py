import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from modules.PredictorAC import PredictorAC
from modules.CLIPEncoder import CLIPEncoder
from modules.VJEPAEncoder import VJEPAEncoder
from modules.MLP import MLP
from src.models.utils.modules import build_action_block_causal_attention_mask

class ActionJEPA(nn.Module):
    def __init__(self, 
                 vjepa_encoder_path, 
                 vjepa_predictor_path,
                 clip_model_path,
                 num_frames=4,
                 vision_dim = 1408,
                 language_dim=768,
                 action_dim = 7, 
                 device="cuda"):
        super(ActionJEPA, self).__init__()

        self.device = device
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.num_frames = num_frames
        
        self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, device=device)
        self.language_backbone = CLIPEncoder(model_path=clip_model_path, device=device)

        self.predictor = PredictorAC(model_path=vjepa_predictor_path, device=device)
        self.T = self.num_frames // 2           
        self.grid_h = 16     
        self.grid_w = 16
        self.cond_tokens = 2 
        mask = build_action_block_causal_attention_mask(
            self.T, self.grid_h, self.grid_w, add_tokens=self.cond_tokens
        )
        self.predictor.predictor.attn_mask = mask.to(device)
        self.predictor.predictor.is_frame_causal = True

        self.language_projector = MLP(input_dim=language_dim, hidden_dims=[1024], output_dim=vision_dim)
        self.actor = MLP(input_dim=vision_dim*2, hidden_dims=[1024, 512, 256], output_dim=action_dim)
        self.refiner = MLP(input_dim=vision_dim*3, hidden_dims=[1024, 512, 256], output_dim=action_dim)

    def forward(self, text_input, vision_input):
        
        if torch.is_tensor(vision_input) and vision_input.dim() == 3:
            z_obs = vision_input
        else:
            z_frames = self.vision_backbone.preprocess_frames(vision_input)
            print(f"z_frames: {z_frames.shape}")
            z_obs = self.vision_backbone(z_frames)
            print(f"z_obs: {z_obs.shape}")
        
        B, N, D = z_obs.shape
      
        if torch.is_tensor(text_input) and text_input.dim() == 3:
            z_text = text_input
        else:
            z_tokens = self.language_backbone.tokenization(text_input)
            print(f"z_tokens: {z_tokens}")
            z_text = self.language_backbone(z_tokens)
            print(f"z_text: {z_text.shape}")

        # Computing mean values to pooling in the sequence of tokens in 1 token
        #z_obs_mean = z_obs.mean(dim=1)
        #print(f"z_obs_mean: {z_obs_mean.shape}")
        z_text_mean = z_text.mean(dim=1)
        print(f"z_text_mean: {z_text_mean.shape}")
        
        # Project the text mean token to the same dimensionality of the V JEPA 2 Encoding 
        z_text_projected = self.language_projector(z_text_mean)
        print(f"z_text_projected: {z_text_projected.shape}")

        z_obs_t = z_obs.view(B, self.T, 256, D)
        print(f"z_obs_t: {z_obs_t.shape}")
        actor_actions_list = []
        for t in range(self.T):
            z_obs_t_mean = z_obs_t[:, t, :, :].mean(dim=1)
            print(f"z_obs_t_mean: {z_obs_t_mean.shape}")
            actor_input_t = torch.cat([z_obs_t_mean, z_text_projected], dim=-1)
            print(f"actor_input: {actor_input_t.shape}")
            actor_action_t = self.actor(actor_input_t)
            print(f"a_actor: {actor_action_t.shape}")
            actor_actions_list.append(actor_action_t)

        # Passing the input to the actor
        
        a_actor_seq = torch.stack(actor_actions_list, dim=1)
        print(f"a_actor_seq: {a_actor_seq.shape}")
        actor_action = a_actor_seq[:, -1, :]

        z_pred_tokens, _, _ = self.predictor(z_obs, a_actor_seq)
        print(f"z_pred_tokens: {z_pred_tokens.shape}")
        z_pred_mean = z_pred_tokens.mean(dim=1)
        print(f"z_pred_mean: {z_pred_mean.shape}")
        
        z_obs_final = z_obs[:, -256:, :].mean(dim=1)
        print(f"z_obs_final: {z_obs_final.shape}")
        z_pred_final = z_pred_tokens[:, -256:, :].mean(dim=1)
        print(f"z_pred_final: {z_pred_final.shape}")
        
        # Creating the input for the refiner as obs, text and prediction
        refiner_input = torch.cat([z_obs_final, z_text_projected, z_pred_final], dim=-1)
        print(f"refiner_input: {refiner_input.shape}")
        refiner_action = self.refiner(refiner_input)
        print(f"refiner_action: {refiner_action.shape}")

        return actor_action, refiner_action
            
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vjepa_path = "checkpoints/facebook/vjepa2-vitg-fpc64-256"
    vjepa_pred_path = "checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar"
    clip_path = "checkpoints/openai/clip-vit-large-patch14"

    model = ActionJEPA(
        vjepa_encoder_path=vjepa_path,
        vjepa_predictor_path=vjepa_pred_path,
        clip_model_path=clip_path,
        num_frames=4,
        device=device
    ).to(device)

    # --- TEST 1: MODO INFERENZA (Dati Grezzi) ---
    # Simuliamo 6 frame (H=224, W=224) e una stringa di testo
    raw_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
    raw_text = "Pick up the red cube"

    # Forward pass
    with torch.no_grad():
        a1_inf, a2_inf = model(raw_text, raw_frames)
    
    print(f"Actor: {a1_inf.shape}")   
    print(f"Refiner: {a2_inf.shape}") 

    z_obs_batch = torch.randn(1, 512, 1408).to(device)
    z_text_batch = torch.randn(1, 77, 768).to(device)

    a1_train, a2_train = model(z_text_batch, z_obs_batch)

    print(f"Batch Actor: {a1_train.shape}")   
    print(f"Batch Refiner: {a2_train.shape}") 

  
