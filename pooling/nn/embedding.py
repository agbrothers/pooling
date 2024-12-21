import torch
import torch.nn as nn
from torch import Tensor


class MutableEmbedding(nn.Embedding):
    '''
    DESC:
    Embedding Subclass that allows for additional 
    embeddings to be added post initialization. 

    '''

    def __init__(
            self,
            num_embd:int, 
            dim_embd:int, 
            idx_pad:int=None, 
            max_norm:int=None, 
            norm_type:float=2, 
            scale_grad_by_freq=False, 
            sparse:bool=False, 
            weight:Tensor=None,
            device=None, 
            dtype=None
        ) -> None:
        super().__init__(num_embd, dim_embd, idx_pad, max_norm, norm_type, scale_grad_by_freq, sparse, weight, device, dtype)
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def add_embd(self, num_embd: int):
        ## Add new embeddings as zero vectors
        ## Should they be standard normal?
        weight = nn.functional.pad(self.weight, (0,0,0,num_embd), value=0)
        self.weight = nn.Parameter(weight.clone(), requires_grad=True)
        self.num_embeddings += num_embd

    def forward(self, input):
        ## A new embedding is encountered
        num_new = torch.max(input).item()
        if num_new > self.num_embeddings:
            self.add_embd(num_new-self.num_embeddings)
        return super().forward(input)
    

class EmbeddingConcatLayer(nn.Module): 
    '''
    CONCATENATE TOKENS WITH LEARNED EMBEDDINGS AND GENERATE MASKS

    '''
    def __init__(
            self,
            dim_token,
            dim_embd=32,
            num_embd=16,
            idx_embd=None,
        ):
        super().__init__()
        
        ## FEATURE INDEX INDICATING WHETHER OR NOT TO MASK A TOKEN
        self._dim_input = dim_token
        self._idx_embd = idx_embd

        ## OPTIONALLY PULL THE TOKEN TYPE INTEGER FROM EACH TOKEN AND USE IT TO LOOK UP LEARNED TYPE EMBEDDINGS
        if self._idx_embd is not None:
            self._dim_input = dim_token - 1 + dim_embd
            self._idx_embd = idx_embd if idx_embd >= 0 else idx_embd + dim_token
            self.embd = MutableEmbedding(num_embd, dim_embd)
            assert self._idx_embd < dim_token and self._idx_embd >= 0, \
                f"The embedding index `{idx_embd}` is out of bounds for observed tokens of size `{dim_token}`"
        return 

    def forward(self, tokens:Tensor):

        ## RETURN INPUT TOKENS
        if self._idx_embd is None:
            return tokens
        
        ## LOOKUP EMBEDDING FROM FEATURE
        int_embd = tokens[..., self._idx_embd]
        embd = self.embd(int_embd.long())

        ## TRUNCATE EMBD IDX FEATURE AND CONCAT EMBEDDING VECTOR
        return torch.cat((
            tokens[..., :self._idx_embd   ], 
            tokens[..., self._idx_embd+1: ],
            embd,
        ), dim=-1)
