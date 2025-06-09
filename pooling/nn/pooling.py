import torch
import torch.nn as nn
from torch import Tensor, BoolTensor

from pooling.nn.attention import MultiHeadAttention


class Pool(nn.Module):
    """
    DESC: 
     Pool Layer Base Class
    
    """
    def __init__(self, dim=-2, **kwargs):
        super().__init__()
        ##  DEFAULT SHAPE ASSUMPTION: 
        ##  [batch, tokens, dim_hidden]
        self.dim = dim

    def forward(self, x:Tensor, mask:BoolTensor=None):
        raise NotImplementedError


class MaxPool(Pool):
    """
    DESC: 
     Take the max value per feature over the embeddings. 
    
    """
    def forward(self, x:Tensor, mask:BoolTensor=None):
        if mask is not None:
            x = x[:, mask]        
        return x.max(dim=self.dim)[0]


class AvgPool(Pool):
    """
    DESC: 
     Take the average value per feature over the embeddings. 
    
    """
    def forward(self, x:Tensor, mask:BoolTensor=None):
        if mask is not None:
            x = x[:, mask]        
        return x.mean(dim=self.dim)


class ClsToken(Pool): 
    """
    DESC: 
     Learn a weighted embedding to be appended to the transformer input,
     then pluck the contextualized embedding from the transformer output.
    
    """
    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, bias=False, cls_token_idx=0, k=1, **kwargs):
        super().__init__(dim=-2)
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias)
        self.cls_token = nn.Parameter(torch.rand(1, k, dim_hidden))
        self.cls_token_idx = cls_token_idx

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        return x[:, self.cls_token_idx]

    def get_cls_token(self, x, **kwargs):
        bs = x.size(0)
        return self.cls_token.expand((bs,-1,-1))


class AdaPool(Pool): 
    """
    DESC: 
     Compute a weighted average per head over the input vectors using attention. 
     By default, one of the input vectors is chosen is used as a query to compute
     the relational weights via dot product with respect to the other input vectos. 
    
    """
    def __init__(self, query_idx=None, **kwargs):
        super().__init__(dim=-2)
        self.query_idx = query_idx
        self.attn = MultiHeadAttention(**kwargs)

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        ## GET QUERY AS A FUNCTION OF THE INPUT SET
        query = self.get_query(x)
        pool = self.attn(query=query, context=x, mask=mask)

        ## IF INDIVIDUAL QUERY, EXTEND SKIP CONNECTION
        if isinstance(self.query_idx, int):
            pool = pool + query
        return pool
    
    def get_query(self, x:Tensor) -> Tensor:
        ## DEFAULT MEAN QUERY
        if self.query_idx is None:
            return torch.mean(x, dim=self.dim, keepdim=False)
        ## INDIVIDUAL QUERY
        elif isinstance(self.query_idx, int):
            return x[:, self.query_idx] 
        ## AGGREGATE QUERY
        elif hasattr(self.query_idx, "__len__"):
            return torch.mean(x[:, self.query_idx], dim=self.dim, keepdim=False)
        else:
            raise NotImplementedError()



if __name__ == "__main__":

    def test_aggr(pool, out_shape, b=32, n=10, d=128):
        x = torch.rand((b, n, d))
        out = pool(x)
        assert len(out.shape) == 2, "Pool error"
        assert out.shape == out_shape, "Pooling hidden dimension size mismatch"
        return 
    
    def test_mask(pool, x, y, mask):
        out = pool(x, mask)
        assert out.shape == y.shape, "Pooling hidden dimension size mismatch"
        assert len(out.shape) == 2, "Pool error"
        assert torch.all(y == out), "Pool Error"
        return         

    ## PARAMETERS
    b = 32 
    n = 10 
    d = dim_hidden = 6
    num_heads = 2
    dropout_w = 0.0
    dropout_e = 0.0
    bias_attn = False

    ## TEST DATA
    a_ = torch.tensor([0, 1, 2, 3, 4, 5])
    b_ = torch.tensor([4, 5, 6, 7, 8, 9])
    x = torch.meshgrid(a_, b_, indexing="xy")[0].T.unsqueeze(0).to(torch.float32)
    mask = torch.BoolTensor([0,1,0,0,1,0])

    ## ASSERT THAT ALL APPROACHES AGGREGATE EMBEDDINGS STACKS TO THE SAME SHAPE
    pooling_layer = MaxPool()
    y = x[:, mask].max(dim=-2)[0]
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    test_mask(pooling_layer, x, y, mask)

    pooling_layer = AvgPool()
    y = x[:, mask].mean(dim=-2)
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    test_mask(pooling_layer, x, y, mask)
    
    pooling_layer = SumPool()
    y = x[:, mask].sum(dim=-2)
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    test_mask(pooling_layer, x, y, mask)
    
    pooling_layer = ClsToken(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, k=1)
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    
    pooling_layer = ClsToken(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, k=3)
    test_aggr(pooling_layer, out_shape=(b, 3*d), b=b, n=n, d=d)
    
    pooling_layer = AdaPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, query_idx=0) 
    torch.nn.init.eye_(pooling_layer.attn.Q.weight)
    torch.nn.init.eye_(pooling_layer.attn.KV.weight[:d])
    torch.nn.init.eye_(pooling_layer.attn.KV.weight[d:])
    torch.nn.init.eye_(pooling_layer.attn.out.weight)
    y = x[:, mask].mean(dim=-2)
    mask = mask.to(torch.float32).masked_fill(mask==True, float('-inf'))    
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    test_mask(pooling_layer, x, y, mask.unsqueeze(0))
    pooling_layer.attn.flash = False
    test_mask(pooling_layer, x, y, mask.unsqueeze(0))

    done = True
