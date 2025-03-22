import math
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from pooling.nn.attention import Attention
# from pooling.nn.gem_attention import GemAttention
from pooling.nn.gem_attention_2 import GemAttention
from pooling.nn.initialize import transformer_init


class FeedForward(nn.Module):
    def __init__(self, dim_hidden, dim_ff, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_ff), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim_ff, dim_hidden), 
            nn.Dropout(dropout), 
        )
        return

    def forward(self, x):
        return self.mlp(x)
    

class TransformerLayer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, num_heads, dropout, gem=False, **kwargs): 
        super().__init__()
        attn_cls = GemAttention if gem else Attention
        self.attn = attn_cls(dim_hidden, num_heads, dropout, **kwargs)
        self.ff = FeedForward(dim_hidden, dim_ff, dropout)
        self.norm_attn = nn.LayerNorm(dim_hidden)
        self.norm_ff   = nn.LayerNorm(dim_hidden)
        return

    def forward(self, x, mask=None):  
        ## SKIP CONNECTION + ATTENTION
        x = x + self.attn(self.norm_attn(x), mask)

        ## SKIP CONNECTION + FEED FORWARD 
        return x + self.ff(self.norm_ff(x))


class Transformer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, num_layers, num_heads, dropout, seed=None, **kwargs) -> None: 
        super().__init__()
        ## SET MODEL PROPERTIES
        self._dim_hidden = dim_hidden
        self._dim_ff = dim_ff
        self._num_layers = num_layers
        self._num_heads = num_heads

        ## INITIALIZE ENCODER
        layer = TransformerLayer(dim_hidden, dim_ff, num_heads, dropout, **kwargs)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim_hidden)

        ## INITIALIZE WEIGHTS
        if seed:
            self.apply(transformer_init)
            for name,param in self.named_parameters():
                if name.endswith("out.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        return

    def forward(self, x, mask=None) -> Tensor:
        for layer in self.layers: 
            x = layer(x, mask=mask)     
        return self.norm(x)


if __name__ == "__main__":

    d = 4

    model = Transformer(
            ## TRANSFORMER PARAMETERS
            dim_hidden=d,
            dim_ff=4*d,
            num_layers=2,
            num_heads=1,
            dropout=0.,
            gem=True,
            flash=False,
            ## GEM ATTENTION EXPERIMENTAL PARAMETERS
            b=0.9,
            p=1,
            p_min=1e-4,
            p_max=5e+4,
            tanh_rate=5e-3,
            eps=1e-10,
            lse=False,
            norm_v=False,
            scale_p=False,
            ## MISC PARAMETERS
            freeze_QKV=True,
            seed=None,
        )
    
    a = torch.tensor([[0, 1, 2, 3, 4, 5]]).T
    X = torch.tile(a, (1, 1, d)).to(torch.float32) + 1
    q = torch.zeros(1,1,d)
    mask = torch.BoolTensor([0,1,0,0,1,0])

    y = torch.std(X, dim=-2)
    # a = model(x=X)

    ## SET WEIGHTS EQUAL TO STD
    # with torch.no_grad():
    #     model.encoder.layers[0].attn.p[:] = 1
    #     model.encoder.layers[1].attn.p[:] = 2
    #     model.encoder.layers[0].attn.Q.weight[:] = 0
    #     model.encoder.layers[1].attn.Q.weight[:] = 0
    #     model.encoder.layers[0].attn.out.weight[:] *= -1

    # b = model(x=X)

    with torch.no_grad():
        model.encoder.layers[1].attn.p[:] = 2
        model.encoder.layers[0].attn.Q.weight[:] = 0
        model.encoder.layers[1].attn.Q.weight[:] = 0
        model.encoder.layers[0].attn.out.weight[:-1] *= 0
        model.encoder.layers[0].attn.out.weight[-1] *= -1
        model.encoder.layers[1].attn.out.weight[:,0] = -1
        model.encoder.layers[1].attn.out.weight[:,1:-1] = 0
        model.encoder.layers[1].attn.out.weight[:,-1] = 1
    
    c = model(x=X)

    catch=True
