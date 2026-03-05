import torch
import os
import numpy as np
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPEncoder(nn.Module):
    def __init__(self, model_path="checkpoints/openai/clip-vit-large-patch14", embedding_dim= 1024 ,device="cpu"):
        super(CLIPEncoder, self).__init__()
        
        # Setting the device (ex. cuda or cpu)
        self.device = device

        # Check if the model was downloaded
        if os.path.exists(model_path): 
            # Setting CLIP tokenizer and encoder
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=model_path).to(device)
        else:
            raise FileNotFoundError(f"CLIP model not found in {model_path}. Run 'python download_models.py' to download it!")

        # Freeze the CLIP text encoder parameters 
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Define a linear layer to project the embedding dimension of CLIP to the same embedding dimension of VJEPA Encoder
        # used to have image and text tokens of the same dimensions
        self.clipencoder_projector = nn.Linear(in_features=self.text_encoder.config.hidden_size, out_features=embedding_dim).to(device)
    
    def forward(self, text):
        text_tokens = self.tokenizer(text, padding=True, truncation=True, max_length = 77, return_tensors="pt").to(self.device)
        print(text_tokens)
        with torch.no_grad():
            outputs = self.text_encoder(**text_tokens)
            last_hidden_state = outputs.last_hidden_state
        
        projected_outputs = self.clipencoder_projector(last_hidden_state)

        return projected_outputs




if __name__ == "__main__":
    clipencoder = CLIPEncoder(device="cuda")
    text = "Pick up the red cube and place it to the right of the blue cube."
    outputs = clipencoder(text)
    print(outputs.shape)