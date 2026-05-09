import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from model.modules.PredictorAC import PredictorAC
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
from model.modules.MLP import MLP
from src.models.utils.modules import build_action_block_causal_attention_mask

class MLPActionJEPA(nn.Module):
    def __init__(self, 
                 vjepa_encoder_path, 
                 vjepa_predictor_path,
                 clip_model_path,
                 num_frames=16,
                 vision_dim = 1408,
                 language_dim=768,
                 action_dim = 7,
                 joint_dim = 7, 
                 embed_dim = 1024,
                 use_backbone = True,
                 frozen_backbone = True,
                 device="cuda"):
        super(MLPActionJEPA, self).__init__()

        self.device = device
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.num_frames = num_frames
        self.use_backbone = use_backbone
        self.frozen_backbone = frozen_backbone
        self.joint_dim = joint_dim
        self.embed_dim = embed_dim
        self.policy = 'mlp'
        
        if self.use_backbone:
            self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, frozen=frozen_backbone, device=device)
            self.language_backbone = CLIPEncoder(model_path=clip_model_path, frozen=frozen_backbone, device=device)
        else:
            self.vision_backbone = None
            self.language_backbone = None

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

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.language_proj= nn.Linear(language_dim, embed_dim)

        self.actor = MLP(input_dim=3*embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim)
        self.refiner = MLP(input_dim=4*embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim)


    def preprocess_frames(self, vision_input):
        z_frames = self.vision_backbone.preprocess_frames(vision_input)
        z_obs = self.vision_backbone(z_frames)
        return z_obs
    
    def preprocess_text(self, language_input):
        z_tokens = self.language_backbone.tokenization(language_input)
        eot_pos = z_tokens['input_ids'].argmax(dim=-1)
        z_text = self.language_backbone(z_tokens)
        return z_text, eot_pos

    def forward(self, language_input, vision_input, joint_input):
            
        if self.use_backbone:
            with torch.no_grad():
                z_obs = self.preprocess_frames(vision_input)
                z_text, eot_pos = self.preprocess_text(language_input)
        else: 
            z_obs = vision_input.to(self.device) if torch.is_tensor(vision_input) else vision_input
            z_text = language_input.to(self.device) if torch.is_tensor(language_input) else language_input

        if joint_input.dim() == 2:
            joint_input = joint_input.unsqueeze(1) # Da [B, 7] a [B, 1, 7]
        
        B, N, D = z_obs.shape   # B = Batch, N = num of tokens, D = dim of each token
        
        eot_embedding = z_text[torch.arange(z_text.shape[0]), eot_pos]
        z_text_proj = self.language_proj(eot_embedding)
        z_joint_proj = self.joint_proj(joint_input)
        z_obs_proj = self.vision_proj(z_obs)


        z_obs_t = z_obs_proj.view(B, self.T, 256, self.embed_dim)
        actor_actions_list = []
        for t in range(self.T):
            z_obs_t_mean = z_obs_t[:, t, :, :].mean(dim=1)
            z_joint_t = z_joint_proj[:,t,:]
            actor_input_t = torch.cat([z_obs_t_mean, z_text_proj, z_joint_t], dim=-1)
            actor_action_t = self.actor(actor_input_t)
            actor_actions_list.append(actor_action_t)

        actor_action_seq = torch.stack(actor_actions_list, dim=1)

        with torch.no_grad():
            z_pred, _, _ = self.predictor(z_obs, actor_action_seq)
        
        z_pred_proj = self.vision_proj(z_pred)
            
        z_pred_t = z_pred_proj.view(B, self.T, 256, self.embed_dim)
        refiner_actions_list = []
        for t in range(self.T):
            z_obs_t_mean = z_obs_t[:, t, :, :].mean(dim=1)
            z_pred_t_mean = z_pred_t[:, t, :, :].mean(dim=1)
            z_joint_t = z_joint_proj[:, t, :]

            refiner_input_t = torch.cat([z_obs_t_mean, z_pred_t_mean, z_text_proj, z_joint_t], dim=-1)
            refiner_action_t = self.refiner(refiner_input_t)
            refiner_actions_list.append(refiner_action_t)
        refiner_action_seq = torch.stack(refiner_actions_list, dim=1)

        return actor_action_seq, refiner_action_seq
    
    def print_model_info(self):
        print("MODEL INFO:\n")
        if self.use_backbone:
            print(f"VISION BACKBONE: {self.vision_backbone.__class__.__name__}\n{self.vision_backbone}")
            print(f"LANGUAGE BACKBONE {self.language_backbone.__class__.__name__}\n{self.language_backbone}")
        print(f"PREDICTOR NETWORK:\n{self.predictor}")
        print(f"ACTOR NETWORK:\n{self.actor}")
        print(f"REFINER NETWORK:\n{self.refiner}")

            
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vjepa_path = "checkpoints/facebook/vjepa2-vitg-fpc64-256"
    vjepa_pred_path = "checkpoints/facebook/jepa-wms/vjepa2_ac_droid.pth.tar/vjepa2_ac_droid.pth.tar"
    clip_path = "checkpoints/openai/clip-vit-large-patch14"

    model = MLPActionJEPA(
        vjepa_encoder_path=vjepa_path,
        vjepa_predictor_path=vjepa_pred_path,
        clip_model_path=clip_path,
        num_frames=2,
        device=device
    ).to(device)

    # --- TEST 1: MODO INFERENZA (Dati Grezzi) ---
    # Simuliamo 6 frame (H=224, W=224) e una stringa di testo
    raw_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    raw_text = "Pick up the red cube"
    joint_input = torch.zeros(1, model.T, 7).to(device)

    # Forward pass
    with torch.no_grad():
        a1_inf, a2_inf = model(raw_text, raw_frames, joint_input)
    
    print(f"Actor: {a1_inf.shape}")   
    print(f"Refiner: {a2_inf.shape}") 

    #z_obs_batch = torch.randn(1, 512, 1408).to(device)
    #z_text_batch = torch.randn(1, 77, 768).to(device)

    #a1_train, a2_train = model(z_text_batch, z_obs_batch)

    #print(f"Batch Actor: {a1_train.shape}")   
    #print(f"Batch Refiner: {a2_train.shape}") 

  
