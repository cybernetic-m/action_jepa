import torch
import torch.nn as nn
import numpy as np


class Predictor(nn.Module):
    def __init__(self, vision_dim = 1280, language_dim = 768 , action_dim = 7, predictor_dim = 1024, depth = 12)