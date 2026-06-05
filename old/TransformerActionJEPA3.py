import torch
import torch.nn as nn
from model.modules.PredictorAC import PredictorAC
from model.modules.CLIPEncoder import CLIPEncoder
from model.modules.VJEPAEncoder import VJEPAEncoder
from model.modules.MLP import MLP

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
                 embed_dim = 1024,
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
        
        # Backbone estrattori
        self.vision_backbone = VJEPAEncoder(model_path=vjepa_encoder_path, frozen=frozen_backbone, device=device)
        self.language_backbone = CLIPEncoder(model_path=clip_model_path, frozen=frozen_backbone, device=device)
        
        # Predittore V-JEPA
        self.predictor = PredictorAC(model_path=vjepa_predictor_path, num_frames=self.num_frames, device=device, finetuned_pred = self.finetuned_pred)
  
        # --- PROIETTORI MLP (Mappatura Non-Lineare) ---
        # Usiamo MLP a 2 livelli con dimensione nascosta intermedia per rendere forte la proiezione
        self.joint_proj = MLP(input_dim=joint_dim, hidden_dims=[256], output_dim=embed_dim, dropout=0.1)
        self.language_proj = MLP(input_dim=language_dim, hidden_dims=[512], output_dim=embed_dim, dropout=0.1)
        self.vision_proj = MLP(input_dim=vision_dim, hidden_dims=[512], output_dim=embed_dim, dropout=0.1)
        self.action_proj = MLP(input_dim=action_dim, hidden_dims=[256], output_dim=embed_dim, dropout=0.1) 

        # Learnable Action Token (Query)
        self.action_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Core Transformers (Decoders)
        self.actor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=16, 
                dim_feedforward=2048, 
                dropout=0.3, 
                batch_first = True),
            num_layers = 24
        )
        
        self.refiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=16, 
                dim_feedforward=2048, 
                dropout=0.3, 
                batch_first = True),
            num_layers = 24
        )

        # Output Heads
        self.actor_head = MLP(input_dim=embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim, dropout=0.2)
        self.refiner_head = MLP(input_dim=embed_dim, hidden_dims=[512, 256, 128], output_dim=action_dim, dropout=0.2)

        #self.apply(self._init_weights)

    def preprocess_frames(self, vision_input):
        z_frames = self.vision_backbone.preprocess_frames(vision_input)
        z_obs = self.vision_backbone(z_frames)
        return z_obs 
    
    def preprocess_text(self, language_input):
        z_tokens = self.language_backbone.tokenization(language_input)
        z_text = self.language_backbone(z_tokens)
        return z_text 
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            if any(name in str(layer) for name in ['actor_head', 'refiner_head']):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)
        elif isinstance(layer, nn.Parameter):
            nn.init.normal_(layer, std=0.02)

    def forward(self, language_input, vision_input, joint_input):
        
        # 1. FEATURE EXTRACTION
        with torch.no_grad():
            z_obs = self.preprocess_frames(vision_input)    # (B, N_patches, vision_dim)
            z_text = self.preprocess_text(language_input)   # (B, seq_len, language_dim)
        
        B, N_patches, _ = z_obs.shape

        # 2. PROIEZIONE TRAMITE I TUOI NUOVI MODULI MLP
        # PyTorch applica l'MLP sull'ultima dimensione preservando (B, N_patches) e (B, seq_len)
        z_obs_proj = self.vision_proj(z_obs)                    # (B, N_patches, embed_dim)
        z_text_proj = self.language_proj(z_text)                # (B, seq_len, embed_dim)
        z_joint_proj = self.joint_proj(joint_input).unsqueeze(1) # (B, 1, embed_dim)

        action_token = self.action_token.expand(B, -1, -1)      # (B, 1, embed_dim)

        # 3. STADIO ACTOR
        actor_context = torch.cat([z_obs_proj, z_text_proj, z_joint_proj], dim=1) # (B, N_patches + seq_len + 1, embed_dim)
        latent_actor_action = self.actor(action_token, actor_context)
        actor_action = self.actor_head(latent_actor_action.squeeze(1)) # (B, action_dim)

        # 4. PROIEZIONE MLP DELL'AZIONE DELL'ACTOR PER IL REFINER
        actor_action_embedded = self.action_proj(actor_action).unsqueeze(1) # (B, 1, embed_dim)
       
        # 5. PREDIZIONE DI JEPA
        with torch.no_grad():
            z_pred, _, _ = self.predictor(z_obs, actor_action.unsqueeze(1)) 
        z_pred_proj = self.vision_proj(z_pred) # (B, N_patches, embed_dim) -> Passa anche lui per l'MLP della visione
        
        # 6. STADIO REFINER
        refiner_context = torch.cat([
            z_obs_proj,             
            z_pred_proj,            
            z_text_proj,            
            z_joint_proj,           
            actor_action_embedded   
        ], dim=1)
        
        latent_refiner_action = self.refiner(action_token, refiner_context)
        refiner_action = self.refiner_head(latent_refiner_action.squeeze(1)) # (B, action_dim)

        return actor_action, refiner_action
    
    def print_model_info(self):
        print("="*60)
        print("                  MODEL ARCHITECTURE INFO                  ")
        print("="*60 + "\n")
        
        print(f"VISION BACKBONE: {self.vision_backbone.__class__.__name__}")
        print(f"LANGUAGE BACKBONE: {self.language_backbone.__class__.__name__}\n")
        
        print(f"ACTION TOKEN SHAPE: {self.action_token.shape}\n")
        
        print(f"LATENT PREDICTOR: {self.predictor.__class__.__name__}\n")
        
        print("-" * 50)
        print("MULTIMODAL PROJECTORS (MLP)")
        print("-" * 50)
        print(f"VISION PROJECTOR:\n{self.vision_proj}")
        print(f"LANGUAGE PROJECTOR:\n{self.language_proj}")
        print(f"JOINT PROJECTOR:\n{self.joint_proj}")
        print(f"ACTION PROJECTOR:\n{self.action_proj}\n")
        
        print("-" * 50)
        print("TRANSFORMER DECODERS")
        print("-" * 50)
        print(f"ACTOR DECODER:\n{self.actor}")
        print(f"REFINER DECODER:\n{self.refiner}\n")
        
        print("-" * 50)
        print("OUTPUT HEADS (MLP)")
        print("-" * 50)
        print(f"ACTOR HEAD:\n{self.actor_head}")
        print(f"REFINER HEAD:\n{self.refiner_head}")
        print("="*60 + "\n")