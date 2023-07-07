import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import DataEmbedding
from attention import ProbAttention, AttentionLayerWin, AttentionLayerCrossWin, FullAttention
from encoder import Encoder, EncoderLayer, ConvEncoderLayer

from decoder import Decoder, DecoderLayerWithFourier

from fourier.module import FourierMix


class PredictorBase(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), window_size=24, num_windows=4, dwindow_size=0):
        super(PredictorBase, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        if dwindow_size == 0:
            dwindow_size = window_size
        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout)
        # Attention

        Attn = ProbAttention if attn == 'prob' else FullAttention
        encoder_layers = nn.ModuleList()

        for l in range(e_layers):
            if l%2 == 0:
                encoder_layers.append(EncoderLayer(
                        AttentionLayerWin(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False,output_attention=output_attention,window_size=window_size),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ))
            else:
                encoder_layers.append(FourierMix(d_model))

        # Encoder
        self.encoder = Encoder(
            encoder_layers,
            [ConvEncoderLayer(d_model) for _ in range(e_layers - 1)]
            if distil
            else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayerWithFourier(
                    AttentionLayerWin(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                        window_size=dwindow_size,
                    ),
                    AttentionLayerCrossWin(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                        num_windows=num_windows,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]