import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log, exp, min, max


class GemAttention(nn.Module):

    def __init__(
            self, 
            dim_hidden, 
            num_heads, 
            dropout=0,  
            flash=True, 
            b=0.9,
            p=0,
            p_min=1e-4,
            p_max=5e+4,
            tanh_rate=5e-3,
            eps=1e-10,
            lse=True,
            norm_v=True,
            activate_v=False,
            scale_p=True,
            freeze_QKV=False,
            squeeze_output=False,
            **kwargs
        ):
        super().__init__()
        self.Q = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.KV = nn.Linear(dim_hidden, 2*dim_hidden, bias=False)
        self.out = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.actv = F.relu
        self.dropout_w = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dim_attn = dim_hidden // num_heads
        self.scale = 1 / math.sqrt(dim_hidden)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash

        ## GeM ATTENTION PARAMETERS
        # assert b > 0 and b < 1, "`b` must fall between 0 and 1."
        self.b = b
        self.p = nn.Parameter(torch.normal(mean=p, std=0.02, size=(dim_hidden,))) ## A
        self.p_max = p_max
        self.p_min = p_min
        self.eps = eps
        self.lse = lse
        self.norm_v = norm_v
        self.activate_v = activate_v
        self.scale_p = scale_p
        self.tanh_rate = tanh_rate
        self.squeeze = squeeze_output

        assert not (norm_v and activate_v), "Must select either v minmax normalization or ReLU activation, not both."

        if freeze_QKV:
            self.freeze_QKV()
        return

    def filter_zero(self, p):
        return p + (p==0) * self.p_min

    def tanh(self, p):
        return self.filter_zero(self.p_max * F.tanh(self.tanh_rate * p) + 1)

    ## [b, 1]
    def norm(self, x, mx, mn):
        return (1 - self.b) * (x- mn) / (mx - mn + self.eps) + self.b
    
    def norm_inv(self, y, mx, mn):
        return (y - self.b) * (mx - mn + self.eps) / (1 - self.b) + mn
    
    # ## POSITIVE SHIFT
    # def norm(self, x, mx, mn):
    #     return (x - mn) + self.b
    
    # def norm_inv(self, y, mx, mn):
    #     return (y - self.b) + mn

    # def f(self, x):
    #     return exp( self.p * log( self.norm(x) ) - self.Z_max )
    
    # def f_inv(self, y):
    #     return self.norm_inv( exp( 1/self.p * (self.Z_max + log(y)) ) )
    
    # def M_f(self, X):
    #     return self.f_inv( torch.mean( self.f(X), axis=-2) )

    def forward(self, context, query=None, mask=None):

        if query is None: query = context

        ## INSTEAD OF EXP/LOG - USE THE FIRST FEW TERMS OF THE TAYLOR EXPANSION FOR SPEED? 
        ## first 3 terms at least

        ## PROJECT INPUTS
        q = self.Q(query)
        k, v = self.KV(context).split(self.dim_hidden, dim=2)

        ## CLAMP AND SHIFT p,v TO PREVENT GeM DISCONTINUITIES
        p = self.tanh(self.p) if self.scale_p else self.p
        if self.norm_v:
            v_max = max(v, dim=-2, keepdim=True).values
            v_min = min(v, dim=-2, keepdim=True).values
            v = self.norm(v, v_max, v_min)
        elif self.activate_v:
            v = self.actv(v) + self.eps

        ## COMPUTE f(v)
        z = p * log(v) 

        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        z = z.view(b, -1, 1, self.num_heads, self.dim_attn).transpose(1, 3)

        ## COMPUTE ATTENTION (WEIGHTED MEAN)
        dot_product = torch.einsum("bhqa,bhka->bhqk", (q, k))
        if mask: dot_product = dot_product.masked_fill_(mask.logical_not(), float("-inf"))
        w = torch.log_softmax(dot_product * self.scale, dim=-1)
        w = self.dropout_w(w)[..., None]
        mean = torch.logsumexp(w + z, dim=-2) 

        ## RESHAPE MEAN
        mean = mean.transpose(1, 2).contiguous().view_as(query) 
        e = exp(mean / p)

        ## COMPUTE f^-1(v)
        if self.norm_v:
            e = self.norm_inv(e, v_max, v_min)
        # assert not torch.any(torch.isnan(e)), "Nans in GeM embeddings"
        # if self.squeeze:
        #     e = e.squeeze(-2)
        return self.dropout_e(self.out(e))

    def freeze_QKV(self):
        torch.nn.init.eye_(self.Q.weight)
        torch.nn.init.eye_(self.KV.weight[:self.dim_hidden])
        torch.nn.init.eye_(self.KV.weight[self.dim_hidden:])
        torch.nn.init.eye_(self.out.weight)
        self.Q.requires_grad_(False)
        self.KV.requires_grad_(False)
        self.KV.requires_grad_(False)
        self.out.requires_grad_(False) 
        return


if __name__ == "__main__":

    def cubic_mean(x, axis=1):
        ## p = 3
        return torch.mean(x**3, axis=axis) ** (1/3)

    ## PARAMETERS
    b = 32 
    n = 10 
    d = dim_hidden = 4
    num_heads = 2

    a = torch.tensor([[0, 1, 2, 3, 4, 5]]).T
    X = torch.tile(a, (1, 1, d)).to(torch.float32) + 1
    q = torch.zeros(1,1,dim_hidden)
    mask = torch.BoolTensor([0,1,0,0,1,0])

    p = torch.arange(0, d, 1, dtype=torch.float32)
    p[0] = -1e+5
    p[-1] = 1e+5

    ## TEST DATA
    # mask = torch.BoolTensor([0,1,0,0,1,0])

    pool = GemAttention(
        d, 
        num_heads, 
        dropout=0., 
        flash=False, 
        freeze_QKV=True, 
        norm_v=False, 
        scale_p=False, 
        squeeze_output=True,
        p=3, 
        b=0.9,
    )

    a = pool(context=X)
    b = cubic_mean(X)
    (X[:,:,0]**3).mean()**(1/3)
