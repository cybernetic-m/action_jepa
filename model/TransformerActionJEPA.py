import torch
import torch.nn as nn
from model.modules.PredictorAC import PredictorAC
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
from model.modules.MLP import MLP
#from src.models.utils.modules import build_action_block_causal_attention_mask

class TransformerActionJEPA(nn.Module):
    def __init__(self, 
                 vjepa_encoder_path, 
                 vjepa_predictor_path,
                 clip_model_path,
                 num_frames=2,
                 vision_dim = 1408,
                 language_dim=768,
                 action_dim = 7,
                 joint_dim = 7, 
                 embed_dim = 1256,
                 frozen_backbone = True,
                 finetuned_pred = False,
                 device="cuda"):
        super(TransformerActionJEPA, self).__init__()

        self.device = device
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.num_frames = num_frames
        self.frozen_backbone = frozen_backbone
        self.joint_dim = joint_dim
        self.embed_dim = embed_dim
        self.policy = 'transformer'
        self.finetuned_pred = finetuned_pred
        
    
        self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, frozen=frozen_backbone, device=device)
        self.language_backbone = CLIPEncoder(model_path=clip_model_path, frozen=frozen_backbone, device=device)
        
        
        self.predictor = PredictorAC(model_path=vjepa_predictor_path, num_frames=self.num_frames, device=device, finetuned_pred = self.finetuned_pred)
        #self.T = self.num_frames // 2           
        #self.grid_h = 16     
        #self.grid_w = 16
        #self.cond_tokens = 2 
        #mask = build_action_block_causal_attention_mask(
        #    self.T, self.grid_h, self.grid_w, add_tokens=self.cond_tokens
        #)
        #self.predictor.predictor.attn_mask = mask.to(device)
        #self.predictor.predictor.is_frame_causal = True

        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.language_proj = nn.Linear(language_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        self.action_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.actor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=2048, 
                dropout=0.2, 
                batch_first = True),
            num_layers = 5
                )
        
        self.refiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=2048, 
                dropout=0.2, 
                batch_first = True),
            num_layers = 5
                )

        self.actor_head = MLP(input_dim=embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim, dropout=0.1)
        self.refiner_head = MLP(input_dim=embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim, dropout=0.1)

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
        
        # FEATURE EXTRACTION
        with torch.no_grad():
            z_obs = self.preprocess_frames(vision_input)
            z_text, eot_pos = self.preprocess_text(language_input)
        
        B, N, D = z_obs.shape   # B = Batch, N = num of tokens, D = dim of each token

        # END OF TOKEN EMBEDDING
        eot_embedding = z_text[torch.arange(B), eot_pos]

        # PROJECTION
        z_obs_proj = self.vision_proj(z_obs)
        z_text_proj = self.language_proj(eot_embedding).unsqueeze(1)
        z_joint_proj = self.joint_proj(joint_input).unsqueeze(1)

        # batching the action token
        action_token = self.action_token.expand(B, -1, -1)

        # ATTENTION TEXT*PATCHES
        attn_logits = torch.bmm(z_obs_proj, z_text_proj.transpose(1, 2)) 
        attn_weights = torch.softmax(attn_logits, dim=1)
        z_obs_attention = torch.sum(z_obs_proj * attn_weights, dim=1, keepdim=True) 

        # ACTOR 
        actor_context = torch.cat([z_obs_attention, z_text_proj, z_joint_proj], dim=1)
        latent_actor_action = self.actor(action_token, actor_context)
        actor_action = self.actor_head(latent_actor_action.squeeze(1))
       
        # PREDICTOR
        with torch.no_grad():
            z_pred, _, _ = self.predictor(z_obs, actor_action.unsqueeze(1))
        z_pred_proj = self.vision_proj(z_pred)
        z_pred_attention = torch.sum(z_pred_proj * attn_weights, dim=1, keepdim=True)
        
        # REFINER
        refiner_context = torch.cat([z_obs_attention, z_pred_attention, z_text_proj, z_joint_proj], dim=1)
        latent_refiner_action = self.refiner(action_token, refiner_context)
        refiner_action = self.refiner_head(latent_refiner_action.squeeze(1))

        return actor_action, refiner_action
    
    def print_model_info(self):
        print("MODEL INFO:\n")
        print(f"VISION BACKBONE: {self.vision_backbone.__class__.__name__}\n{self.vision_backbone}")
        print(f"LANGUAGE BACKBONE {self.language_backbone.__class__.__name__}\n{self.language_backbone}")
        print(f"LEARNABLE ACTION TOKEN:\n{self.action_token.shape}")
        print(f"PREDICTOR LAYERS:\n{self.predictor.__class__.__name__}\n{self.predictor}")
        print(f"LANGUAGE PROJECTOR LAYERS:\n{self.language_proj}")
        print(f"VISION PROJECTOR LAYERS:\n{self.vision_proj}")
        print(f"JOINT PROJECTOR LAYERS:\n{self.joint_proj}")
        print(f"ACTOR LAYERS:\n{self.actor}")
        print(f"REFINER LAYERS:\n{self.refiner}")
        print(f"ACTOR HEAD LAYERS:\n{self.actor_head}")
        print(f"REFINER HEAD LAYERS:\n{self.refiner_head}")