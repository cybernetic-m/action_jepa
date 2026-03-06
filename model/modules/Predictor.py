import torch
import torch.nn as nn
import numpy as np


class Predictor(nn.Module):
    def __init__(self, vision_dim = 1280, language_dim = 768 , action_dim = 7, predictor_dim = 1024, num_layers = 12):
        super(Predictor, self).__init__()

        # Define the linear projector to align vision_dim, language_dim and action_dim to the dimensione of predictor
        self.vision_proj = nn.Linear(in_features = vision_dim, out_features = predictor_dim)
        self.language_proj = nn.Linear(in_features = language_dim, out_features = predictor_dim)
        self.action_proj = nn.Linear(in_features = action_dim, out_features = predictor_dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = predictor_dim,
            nhead = 16,
            batch_first = True,
            norm_first = True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer, num_layers = num_layers)

        self.out_proj = nn.Linear(in_features = predictor_dim, out_features = vision_dim)
