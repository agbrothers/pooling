import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log, exp, min, max


FLOAT_MIN=1e-45

class GemAttention(nn.Module):

    def __init__(
            self, 
            dim_hidden, 
            num_heads, 
            dropout_w=0, 
            dropout_e=0, 
            bias=False, 
            flash=True, 
            b=0.9,
            p=1,
            p_min=1e-4,
            p_max=5e+4,
            tanh_rate=5e-3,
            eps=1e-10,
            lse=True,
            norm_v=True,
            scale_p=True,
            freeze_QKV=False,
            squeeze_output=False,
            **kwargs
        ):
        super().__init__()
        self.Q = nn.Linear(dim_hidden, dim_hidden, bias)
        self.KV = nn.Linear(dim_hidden, 2*dim_hidden, bias)
        self.out = nn.Linear(dim_hidden, dim_hidden, bias)
        self.dropout_w = nn.Dropout(dropout_w)
        self.dropout_e = nn.Dropout(dropout_e)
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dim_attn = dim_hidden // num_heads
        self.scale = 1 / math.sqrt(dim_hidden)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash

        ## GeM ATTENTION PARAMETERS
        # assert b > 0 and b < 1, "`b` must fall between 0 and 1."
        self.b = b
        self.p = nn.Parameter(torch.normal(mean=p, std=0.02, size=(dim_hidden,))) ## A
        # self.p = nn.Parameter(torch.normal(mean=p, std=0.0, size=(dim_hidden,))) ## A
        # self.p = nn.Parameter(torch.normal(mean=0, std=0.02, size=(dim_hidden,)))
        self.p_max = p_max
        self.p_min = p_min
        self.eps = eps
        self.lse = lse
        self.norm_v = norm_v
        self.scale_p = scale_p
        self.tanh_rate = tanh_rate
        self.squeeze = squeeze_output

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

        ## PROJECT INPUTS
        q = self.Q(query)
        k, v = self.KV(context).split(self.dim_hidden, dim=2)

        ## CLAMP AND SHIFT p,v TO PREVENT GeM DISCONTINUITIES
        p = self.tanh(self.p) if self.scale_p else self.p
        if self.norm_v:
            v_max = max(v, dim=-2, keepdim=True).values
            v_min = min(v, dim=-2, keepdim=True).values
            v = self.norm(v, v_max, v_min)

        ## COMPUTE f(v)
        if self.lse:
            z = p * log(v) 
            Z_max = z.max(dim=-2, keepdim=True).values
            v = exp(z - Z_max) 
        else:
            v = v ** p

        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)

        ## COMPUTE ATTENTION (WEIGHTED MEAN)
        if self.flash:
            mean = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout_w.p if self.training else 0, 
                is_causal=False,
                scale=self.scale,
            )
        else:
            dot_product = torch.einsum("bhqa,bhka->bhqk", (q, k))
            if mask: dot_product = dot_product.masked_fill_(mask.logical_not(), float("-inf"))
            w = torch.softmax(dot_product * self.scale, dim=-1)
            w = self.dropout_w(w)
            mean = torch.einsum("bhqv,bhva->bhqa", (w, v)) #.transpose(1, 2).contiguous().view_as(query) #.clamp(min=self.float_min) # prevent float underflow

        ## RESHAPE MEAN
        mean = mean.transpose(1, 2).contiguous().view_as(query) #.clamp(min=self.float_min)  # re-assemble all head outputs side by side

        ## COMPUTE f^-1(v)
        e = exp( 1/p * (Z_max + log(mean)) ) if self.lse else mean ** (1/p)
        if self.norm_v:
            e = self.norm_inv(e, v_max, v_min)
        assert not torch.any(torch.isnan(e)), "Nans in GeM embeddings"
        if self.squeeze:
            e = e.squeeze(-2)
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
    d = dim_hidden = 6
    num_heads = 1
    dropout_w = 0.0
    dropout_e = 0.0
    bias_attn = False

    a = torch.tensor([[0, 1, 2, 3, 4, 5]]).T
    X = torch.tile(a, (1, 1, d)).to(torch.float32) + 1
    q = torch.zeros(1,1,dim_hidden)
    mask = torch.BoolTensor([0,1,0,0,1,0])

    p = torch.arange(0, d, 1, dtype=torch.float32)
    p[0] = -1e+5
    p[-1] = 1e+5

    ## TEST DATA
    # mask = torch.BoolTensor([0,1,0,0,1,0])

    pool = GemAttention(d, 
        num_heads, 
        dropout_w, 
        dropout_e, 
        bias=False, 
        flash=True, 
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
