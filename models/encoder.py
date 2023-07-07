import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class ConvEncoderLayer(nn.Module):

    def __init__(self, c_in: int, use_gpt_style: bool=True) -> None:
        super(ConvEncoderLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        if use_gpt_style: 
            self.down_conv = nn.Conv1d(
                in_channels=c_in,
                out_channels=c_in,
                kernel_size=3,
                padding=padding,
                padding_mode='circular',
            )
        else:
            self.down_conv = nn.Linear(c_in, c_in)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int=None, dropout: float=0.1, activation: str="relu") -> None:
        super(EncoderLayer).__init__()

        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    

    def forward(self, x, attn_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x+y), attn


class EncoderStack(nn.Module):
    
    def __init__(self, encoders: nn.Module, inp_lens: float) -> None:
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens
    

    def forward(self, x, attn_mask=None):
        x_stack = []
        attns = []

        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        return (x_stack, attns)