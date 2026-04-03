import torch
import os
import numpy as np
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPEncoder(nn.Module):
    def __init__(self, model_path , frozen = True, device="cpu"):
        super(CLIPEncoder, self).__init__()
        
        # Setting the device (ex. cuda or cpu)
        self.device = device

        self.frozen = frozen

        # Check if the model was downloaded
        if os.path.exists(model_path): 
            # Setting CLIP tokenizer and encoder
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
            self.language_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=model_path).to(device)
        else:
            raise FileNotFoundError(f"CLIP model not found in {model_path}. Run 'python download_models.py' to download it!")

        # Freeze the CLIP text encoder parameters 
        if self.frozen:
            for param in self.language_encoder.parameters():
                param.requires_grad = False
            self.language_encoder.eval()
        else:
            for param in self.language_encoder.parameters():
                param.requires_grad = True
            self.language_encoder.train()

    def tokenization(self, text):
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length = 77, return_tensors="pt").to(self.device)
        return text_tokens
        
    def forward(self, text_tokens):
        if self.frozen:
            with torch.no_grad():
                outputs = self.language_encoder(**text_tokens)
                last_hidden_state = outputs.last_hidden_state
        else: 
            outputs = self.language_encoder(**text_tokens)
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state




if __name__ == "__main__":
    model_path = "checkpoints/openai/clip-vit-large-patch14"
    clipencoder = CLIPEncoder(model_path=model_path, device="cuda")
    text = "Pick up the red cube and place it to the right of the blue cube."
    text_tokens = clipencoder.tokenization(text)
    outputs = clipencoder(text_tokens)
    print(outputs.shape)
