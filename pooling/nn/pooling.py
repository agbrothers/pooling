import torch
import torch.nn as nn
from torch import Tensor, BoolTensor

from pooling.nn.attention import MultiHeadAttention
from pooling.nn.transformer import FeedForward


class Aggregation(nn.Module):
    """
    Aggregation Layer Base Class
    
    """

    def __init__(self, dim=-2, **kwargs):
        super().__init__()
        ##  DEFAULT SHAPE ASSUMPTION: 
        ## [batch, entities, hidden dim]
        self.dim = dim

    def forward(self, x:Tensor, mask:BoolTensor=None):
        raise NotImplementedError


class MaxPool(Aggregation):
    """
    Take the max value per feature over the embeddings. 
    
    """
        
    def forward(self, x:Tensor, mask:BoolTensor=None):
        if mask is not None:
            x = x[:, mask]        
        return x.max(dim=self.dim)[0]


class AvgPool(Aggregation):
    """
    DESC: 
    Take the average value per feature over the embeddings. 
    
    """
            
    def forward(self, x:Tensor, mask:BoolTensor=None):
        if mask is not None:
            x = x[:, mask]        
        return x.mean(dim=self.dim)


class SumPool(Aggregation):
    """
    Take the sum of values per feature over the embeddings. 
    
    """

    def __init__(self, dim_hidden=None, norm=False, **kwargs):
        super().__init__(dim=-2)
        self.norm = nn.LayerNorm(dim_hidden) if norm else None

    def forward(self, x:Tensor, mask:BoolTensor=None):
        if mask is not None:
            x = x[:, mask]
        pool = x.sum(dim=self.dim)
        if self.norm: pool = self.norm(pool)
        return pool


class RelPool(Aggregation): 
    """
    DESC: 
    Compute a weighted average per head over the embeddings using attention. 
    The embedding of the agent's own state is used as a query to learn a 
    relational aggregation over the entity state embeddings. 
    
    """

    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, dropout_ff=0, bias=False, flash=True, query_idx=0, **kwargs):
        super().__init__(dim=-2)
        self.dim_hidden = dim_hidden
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias, flash)
        self.query_idx = query_idx

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        ## AGGREGATE
        query = x[:, self.query_idx] 
        residual = self.attn(query, x, mask)
        return query + residual
        # return residual
    

class LrnPool(Aggregation): 
    """
    DESC: 
    Compute a weighted average per head over the embeddings using attention. 
    A configurable k learned parameter vectors are used as queries and the resulting
    k attended embeddings are flattened and returned. 
    
    """

    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, bias=False, flash=True, query_idx=0, k=1, **kwargs):
        super().__init__(dim=-2)
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias)
        self.query = nn.Parameter(torch.rand(1, k, dim_hidden))
        self.query_idx = query_idx
        self.batch_query = None
        self.k = k

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        return x[:, self.query_idx]

    def get_query(self, x, **kwargs):
        bs = len(x)
        if self.batch_query and len(self.batch_query) == bs:
            return self.batch_query
        return torch.tile(self.query, (bs, 1, 1))


class CtrPool(Aggregation): 
    """
    DESC: 
    Compute a weighted average per head over the embeddings using attention. 
    The centroid of the input embeddings is used as a query to learn an 
    aggregation over the entity state embeddings. 
    
    """

    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, dropout_ff=0, bias=False, flash=True, query_idx=0, **kwargs):
        super().__init__(dim=-2)
        self.dim_hidden = dim_hidden
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias, flash)
        self.query_idx = query_idx

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        ## AGGREGATE
        query = x[:, self.query_idx] 
        residual = self.attn(query, x, mask)
        return query + residual
        # return residual

    def get_query(self, x, **kwargs):
        return torch.mean(x, dim=1, keepdim=True)
    

class RecurrentQueryPool(Aggregation): 
    """
    DESC: 
    Compute a weighted average per head over the embeddings using attention. 
    The previous output embedding is used as the aggregation query for the 
    current set of entity state embeddings, allowing the recurrent hidden 
    state to govern which entities are attended to. 
    
    """    
    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, bias=False, flash=True, query_idx=0, **kwargs):
        super().__init__(dim=-2)
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias)
        self.query_idx = query_idx
        self.query = None 

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        ## AGGREGATE
        query = x[:, self.query_idx] #.unsqueeze(1)
        residual = self.attn(query, x, mask)
        ## UPDATE RECURRENT QUERY
        self.query = query + residual 
        return self.query

    def set_query(self, recurrent_state:torch.Tensor):
        self.query = recurrent_state
    
    def get_query(self, **kwargs):
        ## INITIALIZE NEW STATE or RESET IF BATCH SIZES DIFFER
        assert self.query is not None 
            # self.query = self.get_initial_query()
        return self.query

    def get_initial_query(self):
        ## INITIALIZE EMPTY MEMORY FROM SCRATCH
        param = next(self.parameters())
        return torch.zeros(1, self.attn.dim_hidden, device=param.device, dtype=param.dtype)


class KmeansPool(Aggregation):

    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, dropout_ff=0, bias=False, flash=True, query_idx=0, **kwargs):
        super().__init__(dim=-2)
        self.dim_hidden = dim_hidden
        self.attn = MultiHeadAttention(dim_hidden, num_heads, dropout_w, dropout_e, bias, flash)
        self.query_idx = query_idx
        self.k

    def forward(self, x:Tensor, mask:BoolTensor=None):  
        ## AGGREGATE
        query = x[:, self.query_idx] 
        residual = self.attn(query, x, mask)
        return query + residual
        # return residual

    def get_query(self, x, **kwargs):
        return self.kmeans(x).reshape(x.shape[0], -1)
    
    # def kmeans(self, x):
    #     for k in self.k:
    #         centroid = ...
    

if __name__ == "__main__":

    def test_aggr(pool, out_shape, b=32, n=10, d=128):
        x = torch.rand((b, n, d))
        out = pool(x)
        assert len(out.shape) == 2, "Aggregation error"
        assert out.shape == out_shape, "Pooling hidden dimension size mismatch"
        return 
    
    def test_mask(pool, x, y, mask):
        out = pool(x, mask)
        assert out.shape == y.shape, "Pooling hidden dimension size mismatch"
        assert len(out.shape) == 2, "Aggregation error"
        assert torch.all(y == out), "Aggregation Error"
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
    
    pooling_layer = LrnPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, k=1)
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)
    
    pooling_layer = LrnPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, k=3)
    test_aggr(pooling_layer, out_shape=(b, 3*d), b=b, n=n, d=d)
    
    pooling_layer = RecurrentQueryPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn)
    test_aggr(pooling_layer, out_shape=(b, d), b=b, n=n, d=d)

    pooling_layer = RelPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, query_idx=0) 
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
