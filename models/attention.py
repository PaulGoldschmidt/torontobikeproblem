import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from masking import ProbMask
from math import sqrt 

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

        return context.transpose(2, 1).contiguous(), attn