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
                 language_dim = 768,
                 action_dim = 7,
                 joint_dim = 7, 
                 embed_dim = 1536,
                 transformer_layers = 6,
                 transformer_heads = 8,
                 transformer_ff_dim = 2048,
                 transformer_dropout = 0.1,
                 mlp_hidden_dims = [512, 256, 128],
                 mlp_dropout = 0.1,
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
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout = transformer_dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.T = self.num_frames // 2

    
        self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, frozen=frozen_backbone, device=device)
        self.language_backbone = CLIPEncoder(model_path=clip_model_path, frozen=frozen_backbone, device=device)
        
        self.predictor = PredictorAC(model_path=vjepa_predictor_path, num_frames=self.num_frames, device=device, finetuned_pred = self.finetuned_pred)
  
        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.language_proj = nn.Linear(language_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        self.action_token = nn.Parameter(torch.randn(1, self.T, embed_dim))
        
        self.actor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=self.transformer_heads, 
                dim_feedforward=self.transformer_ff_dim, 
                dropout=self.transformer_dropout, 
                batch_first = True),
            num_layers = self.transformer_layers
                )
        
        self.refiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=self.transformer_heads, 
                dim_feedforward=self.transformer_ff_dim, 
                dropout=self.transformer_dropout, 
                batch_first = True),
            num_layers = self.transformer_layers
                )

        self.actor_head = MLP(input_dim=embed_dim, hidden_dims=self.mlp_hidden_dims, output_dim=action_dim, dropout=self.mlp_dropout)
        self.refiner_head = MLP(input_dim=embed_dim, hidden_dims=self.mlp_hidden_dims, output_dim=action_dim, dropout=self.mlp_dropout)

        #self.apply(self._init_weights)

    def preprocess_frames(self, vision_input):
        z_frames = self.vision_backbone.preprocess_frames(vision_input)
        z_obs = self.vision_backbone(z_frames)
        return z_obs
    
    def preprocess_text(self, language_input):
        z_tokens = self.language_backbone.tokenization(language_input)
        eot_pos = (z_tokens['input_ids'] == self.language_backbone.tokenizer.eos_token_id).int().argmax(dim=-1)
        z_text = self.language_backbone(z_tokens)
        return z_text, eot_pos
    
    def _init_weights(self, layer):

        # Initialize the MLP head weights through HE initialization, while for the projectors we use a normal distribution, all the biases to zero
        # the trunc normal is used also for linear layers inside the transformers actor and refiner
        if isinstance(layer, nn.Linear):
            if any(name in str(layer) for name in ['actor_head', 'refiner_head']):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.trunc_normal_(layer.weight, std=0.02)
            
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        # LayerNorm layers weights will have bias 0 and weights 1
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)
        
        # For the action token we have normally distributed weights with std 0.02
        elif isinstance(layer, nn.Parameter):
            nn.init.normal_(layer, std=0.02)

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
        actor_action = self.actor_head(latent_actor_action.view(B*self.T, -1))
        actor_action = actor_action.view(B, self.T, self.action_dim)
       
        # PREDICTOR
        with torch.no_grad():
            print(z_obs.shape)
            print(actor_action.shape)
            z_pred, _, _ = self.predictor(z_obs, actor_action)
        z_pred_proj = self.vision_proj(z_pred)
        z_pred_attention = torch.sum(z_pred_proj * attn_weights, dim=1, keepdim=True)
        
        # REFINER
        refiner_context = torch.cat([z_obs_attention, z_pred_attention, z_text_proj, z_joint_proj], dim=1)
        latent_refiner_action = self.refiner(action_token, refiner_context)
        refiner_action = self.refiner_head(latent_refiner_action.view(B*self.T, -1))
        refiner_action = refiner_action.view(B, self.T, self.action_dim)

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