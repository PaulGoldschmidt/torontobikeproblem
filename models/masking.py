import torch 


class ProbMask(object):
    
    def __init__(
            self, 
            B: int,
            H: int, 
            L: int, 
            index: torch.Tensor, 
            scores: torch.Tensor, 
            device: str="cpu"
        ) -> None:
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self) -> torch.Tensor:
        return self._mask