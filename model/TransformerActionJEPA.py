import torch
import torch.nn as nn
from model.modules.PredictorAC import PredictorAC
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
from model.modules.MLP import MLP
from src.models.utils.modules import build_action_block_causal_attention_mask

class TransformerActionJEPA(nn.Module):
    def __init__(self, 
                 vjepa_encoder_path, 
                 vjepa_predictor_path,
                 clip_model_path,
                 num_frames=16,
                 vision_dim = 1408,
                 language_dim=768,
                 action_dim = 7,
                 joint_dim = 7, 
                 embed_dim = 512,
                 use_backbone = True,
                 frozen_backbone = True,
                 device="cuda"):
        super(TransformerActionJEPA, self).__init__()

        self.device = device
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.num_frames = num_frames
        self.use_backbone = use_backbone
        self.frozen_backbone = frozen_backbone
        self.joint_dim = joint_dim
        self.embed_dim = embed_dim
        
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

        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.language_proj = nn.Linear(language_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        self.action_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.actor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=2048, 
                dropout=0.1, 
                batch_first = True),
            num_layers = 3
                )
        
        self.refiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=2048, 
                dropout=0.1, 
                batch_first = True),
            num_layers = 3
                )

        self.actor_head = MLP(input_dim=embed_dim, hidden_dims=[256, 128], output_dim=action_dim)
        self.refiner_head = MLP(input_dim=embed_dim, hidden_dims=[256, 128], output_dim=action_dim)

    def forward(self, text_input, vision_input, joint_input):
            
        if self.use_backbone:
            with torch.no_grad():
                z_frames = self.vision_backbone.preprocess_frames(vision_input)
                #print(f"z_frames: {z_frames.shape}")
                z_obs = self.vision_backbone(z_frames)
                #print(f"z_obs: {z_obs.shape}")
        else: 
            z_obs = vision_input.to(self.device) if torch.is_tensor(vision_input) else vision_input
            #print(f"z_obs: {z_obs.shape}")
        
        B, N, D = z_obs.shape   # B = Batch, N = num of tokens, D = dim of each token

        if self.use_backbone:
            with torch.no_grad():
                z_tokens = self.language_backbone.tokenization(text_input)
                #print(f"z_tokens: {z_tokens}")
                z_text = self.language_backbone(z_tokens)
                #print(f"z_text: {z_text.shape}")
        else:
            z_text = text_input.to(self.device) if torch.is_tensor(text_input) else text_input
            #print(f"z_text: {z_text.shape}")

        z_obs_proj = self.vision_proj(z_obs)
        z_text_proj = self.language_proj(z_text)
        z_joint_proj = self.joint_proj(joint_input)

        action_token_batch = self.action_token.expand(B, -1, -1)

        z_obs_t = z_obs_proj.view(B, self.T, 256, self.embed_dim)
        #print(f"z_obs_t: {z_obs_t.shape}")
        actor_actions_list = []
        for t in range(self.T):
            kv_t = z_obs_t[:, t, :, :]
            q_t = torch.cat([z_text_proj, z_joint_proj[:,t:t+1,:], action_token_batch], dim=1)
            #print(f"z_obs_t_mean: {z_obs_t_mean.shape}")
            #print(f"actor_input: {actor_input_t.shape}")
            latent_actor_action_t = self.actor(q_t, kv_t)
            actor_action_t = self.actor_head(latent_actor_action_t[:,-1,:])
            #print(f"actor_action_t: {actor_action_t.shape}")
            actor_actions_list.append(actor_action_t)

        actor_action_seq = torch.stack(actor_actions_list, dim=1)
        #print(f"actor_action_seq: {actor_action_seq.shape}")

        with torch.no_grad():
            z_pred, _, _ = self.predictor(z_obs, actor_action_seq)
            #print(f"z_pred_tokens: {z_pred_tokens.shape}")
            
        z_pred_proj = self.vision_proj(z_pred)
        z_pred_t = z_pred_proj.view(B, self.T, 256, self.embed_dim)
        #print(f"z_obs_t: {z_obs_t.shape}")
        refiner_actions_list = []
        for t in range(self.T):
            kv_t = torch.cat([z_obs_t[:, t, :, :], z_pred_t[:,t,:,:]], dim=1)
            q_t = torch.cat([z_text_proj, z_joint_proj[:,t:t+1,:], action_token_batch], dim=1)
            #print(f"z_obs_t_mean: {z_obs_t_mean.shape}")
            #print(f"actor_input: {actor_input_t.shape}")
            latent_refiner_action_t = self.refiner(q_t, kv_t)
            refiner_action_t = self.refiner_head(latent_refiner_action_t[:,-1,:])
            #print(f"actor_action_t: {actor_action_t.shape}")
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