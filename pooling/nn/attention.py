import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, bias=False, flash=True, **kwargs):
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

    def forward(self, query, context, mask=None):

        ## PROJECT INPUTS
        q = self.Q(query)
        k, v = self.KV(context).split(self.dim_hidden, dim=2)

        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)

        ## COMPUTE ATTENTION
        if self.flash:
            attn_mask = torch.tile(mask, (q.size(-2), 1)) if mask is not None else None 
            e = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_w.p if self.training else 0, is_causal=False)
            e = e.transpose(1, 2).contiguous().view_as(query) 
        else:
            dot_product = torch.einsum("bhqa,bhka->bhqk", (q, k))
            dot_product = self.apply_mask(dot_product * self.scale, mask)
            w = torch.softmax(dot_product, dim=-1)
            w = self.dropout_w(w)
            e = torch.einsum("bhqv,bhva->bhqa", (w, v)).contiguous().view_as(query) 
        
        return self.dropout_e(self.out(e))

    def apply_mask(self, x, mask):
        ## TRUE VALUES GET MASKED
        if mask is None:
            return x
        if len(mask.shape) == 1: 
            mask = mask.unsqueeze(0)
        # if len(mask.shape) == 2: 
        #     mask = mask.unsqueeze(2)
        return x + mask #.masked_fill(mask, float('-inf'))
    