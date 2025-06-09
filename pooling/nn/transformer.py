import math
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from pooling.nn.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, dim_hidden, dim_ff, dropout, bias=False):
        super().__init__()
        self.l_in  = nn.Linear(dim_hidden, dim_ff, bias)
        self.l_out = nn.Linear(dim_ff, dim_hidden, bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l_in(x)
        x = self.gelu(x)
        x = self.l_out(x)
        x = self.dropout(x)
        return x
    

class TransformerLayer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, num_heads, dropout_w=0, dropout_e=0, dropout_ff=0, bias_attn=False, bias_ff=False, flash=True, **kwargs): 
        super().__init__()
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, flash)
        self.ff   = FeedForward(dim_hidden, dim_ff, dropout_ff, bias_ff)
        self.norm_attn = nn.LayerNorm(dim_hidden)
        self.norm_ff   = nn.LayerNorm(dim_hidden)
        self._dim_hidden = dim_hidden
        self._dim_ff = dim_ff

    def forward(self, query, context=None, mask=None):  
        ## MULTIHEAD ATTENTION + SKIP CONNECTION
        if context is None: context = query
        residual = self.attn(self.norm_attn(query), self.norm_attn(context), mask)
        x = query + residual

        ## FEED FORWARD NETWORK + SKIP CONNECTION   
        residual = self.ff(self.norm_ff(x))
        return x + residual


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm_output=False):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer._dim_hidden) if norm_output else None
        self._num_layers = num_layers

    def forward(self, x, mask=None) -> Tensor: 
        for layer in self.layers: 
            x = layer(x, mask=mask)
        if self.norm is not None:  
            ## NOTE: Norm hurts regression performance for all methods            
            x = self.norm(x)
        return x


class Transformer(nn.Module):

    def __init__(
            self, 
            dim_hidden,
            dim_ff,
            num_layers,
            num_heads,
            dropout_w,
            dropout_e,
            dropout_ff,
            bias_attn,
            bias_ff,
            flash,
            norm_output=False,
            seed=None,
            **kwargs,
        ) -> None: 
        super().__init__()

        ## SET MODEL PROPERTIES
        self._dim_hidden = dim_hidden
        self._num_layers = num_layers

        ## INITIALIZE ENCODER
        layer = TransformerLayer(dim_hidden, dim_ff, num_heads, dropout_w, dropout_e, dropout_ff, bias_attn, bias_ff, flash, **kwargs)
        self.encoder = Encoder(layer, num_layers, norm_output)

        ## INITIALIZE WEIGHTS
        if seed:
            self.apply(self.initialize)
            for name,param in self.named_parameters():
                if name.endswith("out.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        return


    def forward(self, x, mask=None) -> Tensor:
        return self.encoder(x, mask=mask)


    def initialize(self, module) -> None:
        """ 
        INITIALIZATION SCHEME AS IN 
        [1] https://arxiv.org/pdf/1502.01852.pdf
        [2] https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L163
        
        """
        ## LINEAR LAYERS
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        ## LAYERNORMS
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        ## EMBEDDING WEIGHTS
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        return
