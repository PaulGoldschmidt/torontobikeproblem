import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from masking import ProbMask
from math import sqrt 


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):

    def __init__(
            self, 
            mask_flag: bool=True, 
            factor: int=5, 
            scale: torch.Tensor=None, 
            attention_dropout: float=0.1, 
            output_attention: bool=False,
        ) -> None:
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _compute_prob_qk(
            self, 
            Q: torch.Tensor, 
            K: torch.Tensor, 
            sample_k: int, 
            n_top: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        # print(index_sample.shape)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)

        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  
        return (Q_K, M_top)
    
    def _get_initial_context(
            self, 
            V: torch.Tensor, 
            L_Q: torch.Tensor,
        ) -> torch.Tensor:
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            return V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert(L_Q == L_V)
            return V.cumsum(dim=-2)

    def _update_context(
            self, 
            context_in: torch.Tensor, 
            V: torch.Tensor,  
            scores: torch.Tensor,  
            index: torch.Tensor, 
            L_Q: int, 
            attn_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
    
    def forward(
            self, 
            queries: torch.Tensor, 
            keys: torch.Tensor, 
            values: torch.Tensor, 
            attn_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                    np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
                    np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._compute_prob_qk(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)

        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return (context.transpose(2, 1).contiguous(), attn)
    

class AttentionLayer(nn.Module):

    def __init__(
            self, 
            attention: nn.Module,
            d_model: int, 
            n_heads: int,
            d_keys: int=None, 
            d_values: int=None, 
            mix: bool=False, 
            layer_num: int=0,
        ) -> None:
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num

    def forward(
            self, 
            queries: torch.Tensor, 
            keys: torch.Tensor, 
            values: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return (self.out_projection(out), attn)


class AttentionLayerWin(nn.Module):
    
    def __init__(
            self, 
            attention: nn.Module, 
            d_model: int, 
            n_heads: int,     
            d_keys: int=None, 
            d_values: int=None, 
            mix: bool=False, 
            layer_num: int=0, 
            window_size: int=8,
            output_attention: bool=False
        ) -> None:
        super(AttentionLayerWin, self).__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num
        self.window_size = window_size
        self.output_attn = output_attention
    
    def forward(
            self, 
            queries: torch.Tensor, 
            keys: torch.Tensor, 
            values: torch.Tensor, 
            attn_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        #Partition the vectors into windows
        queries = queries.view(B*(L//self.window_size), self.window_size, H, -1)
        keys = keys.view(B*(S//self.window_size), self.window_size, H, -1)
        values = values.view(B*(S//self.window_size), self.window_size, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.output_attn:
            attn = self._output_attn(L, attn)

        out = out.view(B, L, H, -1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return (self.out_projection(out), attn)

    def _output_attn(self, L: int, attn: torch.Tensor) -> torch.Tensor:
        num_window = L//self.window_size

        for k in range(num_window):
            if k==0:
                p2d = (0,((num_window-(k+1))*self.window_size))
                attn_tmp = F.pad(attn[:self.window_size,:,:,:],p2d)
            else:
                p2d = (k*self.window_size, (num_window-(k+1))*self.window_size)
                attn_tmp = torch.cat((attn_tmp, F.pad(attn[k*self.window_size:(k+1)*self.window_size,:,:,:],p2d)),dim=2)

        return attn_tmp


class AttentionLayerCrossWin(nn.Module):
    
    def __init__(
            self, 
            attention: nn.Module, 
            d_model: int, 
            n_heads: int,
            d_keys: int=None, 
            d_values: int=None, 
            mix: bool=False, 
            layer_num: int=0, 
            num_windows: int=4, 
            output_attention: bool=False
        ) -> None:
        super(AttentionLayerCrossWin, self).__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num
        self.num_windows = num_windows
        self.output_attn = output_attention
    
    def forward(
            self, 
            queries: torch.Tensor, 
            keys: torch.Tensor, 
            values: torch.Tensor, 
            attn_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        #Partition the vectors into windows
        queries = queries.view(B*self.num_windows, L//self.num_windows, H, -1)
        keys = keys.view(B*self.num_windows, S//self.num_windows, H, -1)
        values = values.view(B*self.num_windows, S//self.num_windows, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.output_attn:
            attn = self._output_attn(L, attn)

        out = out.view(B, L, H, -1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def _output_attn(self, L: int, attn: torch.Tensor) -> torch.Tensor:
        window_size = L//self.num_windows

        for k in range(self.num_window):
            if k==0:
                p2d = (0,((self.num_windows-(k+1))*window_size))
                attn_tmp = F.pad(attn[:window_size,:,:,:],p2d)
            else:
                p2d = (k*window_size, (self.num_windows-(k+1))*window_size)
                attn_tmp = torch.cat((attn_tmp, F.pad(attn[k*window_size:(k+1)*window_size,:,:,:],p2d)),dim=2)

        return attn_tmp