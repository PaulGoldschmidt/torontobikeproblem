import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FourierLayer(nn.Module):
    def __init__(self, seq_dim=1, hidden_dim=2):
        super().__init__()
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim
    def forward(self, x):
        B, L, D = x.shape
        return torch.real(torch.fft.fft(torch.fft.fft(x,dim=self.hidden_dim), dim=self.seq_dim))


class FourierMix(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fourier = FourierLayer(1,2)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x,  attn_mask=None):
        x = self.fourier(x)
        x = self.norm(x)
        return x, None


class FeedFoward(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation="relu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.dropout(self.fc2(x))
        return x