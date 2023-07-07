import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from module import FeedFoward, FourierLayer


class FNetLayer(nn.Module):
   
    def __init__(self, d_model, dropout=0.1, activation="relu"):
        super().__init__()
        self.fourier = FourierLayer(1,2)
        self.feedforward = FeedFoward(d_model, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
   
    def forward(self, x, attn_mask=None):
        new_x = self.fourier(x)
        x = x + self.dropout(new_x)

        x = self.norm1(x)
        x = x + self.feedforward(x)
        return self.norm2(x), None
