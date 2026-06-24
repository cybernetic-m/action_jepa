import torch
import torch.nn as nn
from model.modules.PredictorAC import PredictorAC
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
from model.modules.MLP import MLP
#from src.models.utils.modules import build_action_block_causal_attention_mask
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # register_buffer salva il tensore nello stato del modello senza renderlo addestrabile (no gradienti)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Somma il positional encoding sinusoidale alla sequenza di input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TransformerActionJEPA(nn.Module):
    def __init__(self, 
                 vjepa_encoder_path, 
                 vjepa_predictor_path,
                 clip_model_path,
                 num_frames=2,
                 action_chunk_size = 10,
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
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout = transformer_dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.action_chunk_size = action_chunk_size
        

        self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, frozen=frozen_backbone, device=device)
        self.language_backbone = CLIPEncoder(model_path=clip_model_path, frozen=frozen_backbone, device=device)
        
        self.predictor = PredictorAC(model_path=vjepa_predictor_path, num_frames=self.num_frames, frozen=False, device=device)
  
        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.language_proj = nn.Linear(language_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        self.action_token = nn.Parameter(torch.randn(1, self.action_chunk_size, embed_dim))
        
        self.pos_embedding = PositionalEncoding(d_model=embed_dim, dropout=0.0, max_len=1000)

        self.multimodal_contextualizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=self.transformer_heads, 
                dim_feedforward=self.transformer_ff_dim, 
                dropout=self.transformer_dropout, 
                batch_first=True
            ),
            num_layers=self.transformer_layers
        )
        
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
            z_text, _ = self.preprocess_text(language_input)
        
        B, N, D = z_obs.shape   # B = Batch, N = num of tokens, D = dim of each token

        # PROJECTION
        z_obs_proj = self.vision_proj(z_obs)
        z_text_proj = self.language_proj(z_text)
        z_joint_proj = self.joint_proj(joint_input).unsqueeze(1)

        # batching the action token
        action_token = self.action_token.expand(B, -1, -1)

        # CONTEXTUALIZER FOR ACTOR
        encoder_input = torch.cat([z_joint_proj, z_text_proj, z_obs_proj], dim=1)
        encoder_input = self.pos_embedding(encoder_input) 
        actor_context = self.multimodal_contextualizer(encoder_input)

        # ACTOR 
        latent_actor_action = self.actor(action_token, actor_context)
        actor_action = self.actor_head(latent_actor_action.view(B*self.action_chunk_size, -1))
        actor_action = actor_action.view(B, self.action_chunk_size, self.action_dim)

        # PREDICTOR
        actor_action_for_predictor = actor_action[:, 0:1, :]
        z_pred, _, _ = self.predictor(z_obs[:,-256:,:], actor_action_for_predictor)
        z_pred_proj = self.vision_proj(z_pred)
        
        # CONTEXTUALIZER FOR REFINER
        encoder_input = torch.cat([z_joint_proj, z_text_proj, z_obs_proj, z_pred_proj], dim=1)
        encoder_input = self.pos_embedding(encoder_input)        
        refiner_context = self.multimodal_contextualizer(encoder_input)

        # REFINER
        latent_refiner_action = self.refiner(action_token, refiner_context)
        refiner_action = self.refiner_head(latent_refiner_action.view(B*self.action_chunk_size, -1))
        refiner_action = refiner_action.view(B, self.action_chunk_size, self.action_dim)

        return actor_action, refiner_action, z_pred
    
    def print_model_info(self):
        print("MODEL INFO:\n")
        print(f"VISION BACKBONE: {self.vision_backbone.__class__.__name__}\n{self.vision_backbone}")
        print(f"LANGUAGE BACKBONE {self.language_backbone.__class__.__name__}\n{self.language_backbone}")
        print(f"LEARNABLE ACTION TOKEN:\n{self.action_token.shape}")
        print(f"PREDICTOR LAYERS:\n{self.predictor.__class__.__name__}\n{self.predictor}")
        print(f"LANGUAGE PROJECTOR LAYERS:\n{self.language_proj}")
        print(f"VISION PROJECTOR LAYERS:\n{self.vision_proj}")
        print(f"JOINT PROJECTOR LAYERS:\n{self.joint_proj}")
        print(f"MULTIMODAL CONTEXTUALIZER:\n{self.multimodal_contextualizer}")
        print(f"ACTOR LAYERS:\n{self.actor}")
        print(f"REFINER LAYERS:\n{self.refiner}")
        print(f"ACTOR HEAD LAYERS:\n{self.actor_head}")
        print(f"REFINER HEAD LAYERS:\n{self.refiner_head}")