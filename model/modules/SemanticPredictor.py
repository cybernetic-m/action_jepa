import torch
import torch.nn as nn
import numpy as np


class SemanticPredictor(nn.Module):
    def __init__(self, 
                 vision_tokens = 2048, 
                 language_tokens = 77, 
                 state_tokens = 1, 
                 action_tokens = 1,
                 vision_dim = 1280, 
                 language_dim = 768 , 
                 action_dim = 7, 
                 state_dim=7, 
                 predictor_dim = 1024, 
                 num_layers = 12):
        super(SemanticPredictor, self).__init__()

        # Define the linear projector to align vision_dim, language_dim, action_dim and state_dim to the dimensione of predictor
        self.vision_proj = nn.Linear(in_features = vision_dim, out_features = predictor_dim)
        self.language_proj = nn.Linear(in_features = language_dim, out_features = predictor_dim)
        self.action_proj = nn.Linear(in_features = action_dim, out_features = predictor_dim)
        self.state_proj = nn.Linear(in_features = state_dim, out_features = predictor_dim)

        # Define the transformer encoder layer that take in input a token sequence where each token embedding dim is "predictor_dim"
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = predictor_dim,
            nhead = 16,
            batch_first = True,
            norm_first = True
        )

        # Total number of tokens in the sequence in input to the transformer encoder
        self.num_tokens = vision_tokens + language_tokens + state_tokens + action_tokens

        # Defining the learnable positional embedding vectors to sum to each token in the sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, 0.02) # Initialize the pos_embed numbers with a gaussian distribution with zero mean and 0.02 standard deviation

        # Defining the entire transformer encoder with "num_layers" transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer, num_layers = num_layers)

        # Defining the linear projector to reproject the output of the last transformer encoder layer back to vision_dim (the dimension of V-JEPA Backbone)
        self.out_proj = nn.Linear(in_features = predictor_dim, out_features = vision_dim)

    def forward(self, z_goal, state, action, z_obs):

        # Save the dimension of how much vision tokens we have in input because we need to extract from out_proj
        num_vision_tokens = z_obs.shape[1]
        
        # Projecting inputs to the prediction_dim feature space
        goal_feat = self.language_proj(z_goal)
        state_feat = self.state_proj(state).unsqueeze(1) # state is a tensor [Batch_size,7], we need unsqueeze to add a dimension [Batch_size,1,7]
        action_feat = self.action_proj(action).unsqueeze(1) # action is a tensor [Batch_size,7], we need unsqueeze to add a dimension [Batch_size,1,7]
        vision_feat = self.vision_proj(z_obs)

        # Concatenate to create the x token sequence
        x = torch.cat([goal_feat, state_feat, action_feat, vision_feat], dim=1)

        # Summing the input x with positional embedding learnable vectors
        x = x + self.pos_embed

        # Passing the x token sequence to the transformer encoder block
        x = self.transformer(x)

        # Returning to the dimension of vision_dim (V-JEPA dimension for self-supervision loss)
        # Returning only the prediction of the vision tokens, to compare to the ground truth vision tokens from the video
        z_next_pred = self.out_proj(x[:, -num_vision_tokens:, :])

        return z_next_pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = Predictor().to(device)

    z_obs = torch.randn(1, 2048, 1280).to(device)
    z_goal = torch.randn(1, 77, 768).to(device)
    state = torch.randn(1, 7).to(device)
    action = torch.randn(1, 7).to(device)

    z_next_pred = predictor(z_goal, state, action, z_obs)
    print(z_next_pred.shape)
        
        

