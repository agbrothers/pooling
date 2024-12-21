import math
import torch
import torch.nn as nn
from torch import Tensor

from pooling.nn.transformer import Transformer
from pooling.nn.pooling import (
    MaxPool, 
    AvgPool, 
    SumPool, 
    RelPool, 
    LrnPool, 
    CtrPool,
    RecurrentQueryPool,
)


class Attenuator(nn.Module):

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
            dim_input=None,
            dim_output=None,
            flash=True,
            num_emb=2,            
            pos_emb=False,
            pooling_norm=False,
            pooling_method="relational_query",
            **kwargs,
        ) -> None: 
        super().__init__()

        ## SET MODEL PROPERTIES
        self._dim_hidden = dim_hidden
        self._dim_ff = dim_ff
        self._dim_input = dim_input
        self._dim_output = dim_output
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_w = dropout_w,
        self._dropout_e = dropout_e,
        self._dropout_ff = dropout_ff,
        self._bias_attn = bias_attn,
        self._bias_ff = bias_ff,
        self._pooling_norm = pooling_norm
        self._pooling_method = pooling_method
        self._flash = flash
        self._query = False
        self._recurrent = False

        ## INITIALIE ENCODER
        self.transformer = Transformer(
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
            **kwargs,
        )
        self.proj_in = None
        if dim_input:
            self.proj_in = nn.Linear(dim_input, dim_hidden, bias=bias_ff)

        self.proj_out = None
        if dim_output:
            self.proj_out = nn.Linear(dim_hidden, dim_output, bias=bias_ff)

        self.pos_emb = None
        if pos_emb:
            self.pos_emb = nn.Embedding(num_embeddings=num_emb, embedding_dim=dim_hidden)

        ## INITIALIZE POOLING LAYER
        if pooling_method == "MaxPool":
            pooling_layer = MaxPool()
        elif pooling_method == "AvgPool":
            pooling_layer = AvgPool()
        elif pooling_method == "SumPool":
            pooling_layer = SumPool(dim_hidden, pooling_norm)
        elif pooling_method == "RelPool":
            pooling_layer = RelPool(dim_hidden, num_heads, dropout_w, dropout_e, dropout_ff, bias_attn, flash, query_idx=0) 
            self.pos_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_hidden)
            # self._query = True
        elif pooling_method == "LrnPool":
            pooling_layer = LrnPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, flash, k=1)
            # self.pos_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_hidden)
            self._query = True
        elif pooling_method == "CtrPool":
            pooling_layer = CtrPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, flash, k=1)
            self.pos_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_hidden)
            self._query = True
        # elif pooling_method == "learned_queries":
        #     pooling_layer = LrnPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, flash, k=3)
        # elif pooling_method == "recurrent_query":
        #     pooling_layer = RecurrentQueryPool(dim_hidden, num_heads, dropout_w, dropout_e, bias_attn, flash, query_idx=0)
        #     self._recurrent = True
        #     self._query = True
        else:
            raise ValueError("Invalid `pooling_method` argument for AttenuationNetwork.")

        self.pool = pooling_layer

        ## INITIALIZE WEIGHTS
        self.apply(self.initialize)
        # for name,param in self.named_parameters():
        # #     if name.endswith("out.weight"):
        # #         torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        #     if name.endswith("pool.attn.Q.weight"):
        #         torch.nn.init.normal_(param, mean=0.0, std=0.0000002)
        #     if name.endswith("pool.attn.KV.weight"):
        #         # with torch.no_grad():
        #         #     param[:dim_hidden] = torch.nn.Parameter(-self.pool.attn.Q.weight.clone().detach())
        #         # torch.nn.init.normal_(param[:dim_hidden], mean=0.0, std=0.0002)
        #         # torch.nn.init.eye_(param[:dim_hidden])
        #         # torch.nn.init.eye_(param[:dim_hidden])
        #         torch.nn.init.eye_(param[dim_hidden:])
        #     elif name.endswith("pool.attn.out.weight"):
        #         torch.nn.init.eye_(param)
        return


    def forward(self, x, mask=None) -> Tensor:
        ## PROJECT INPUT TO HIDDEN DIMENSION
        if self.proj_in:
            x = self.proj_in(x)
        
        ## SOURCE QUERY VECTOR
        if self._query:
            x = torch.cat((self.pool.get_query(x), x), dim=1)
            ## TODO: IMPLEMENT MASK EXTENSION
            # if mask is not None:
            #     mask = torch.cat((torch.ones(mask.shape[1]), mask), dim=1)
        
        ## ADD POSITIONAL EMBEDDINGS
        if self.pos_emb:
            idx = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
            idx[:, 0] = 0
            x = x + self.pos_emb(idx)

        ## STEP TRANSFORMER ENCODER
        x = self.transformer(x, mask)

        ## APPLY POOLING
        pool = self.pool(x, mask)

        ## PROJECT HIDDEN TO OUTPUT DIMENSION
        if self.proj_out:
            pool = self.proj_out(pool)
        return pool


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


    def get_query(self):
        return self.pool.get_query()
    
    def set_recurrent_state(self, recurrent_state):
        ## OVERRIDE THE RECURRENT STATE (RNNs ONLY)
        self.pool.set_query(recurrent_state)

    def get_recurrent_state(self):
        ## RETURN THE RECURRENT STATE (RNNs ONLY)
        return self.pool.get_query()

    def get_initial_recurrent_state(self):
        ## RETURN AN INITIAL RECURRENT STATE (RNNs ONLY)
        return self.pool.get_initial_query()

    def get_recurrent_state_shape(self):
        ## RETURN THE RECURRENT STATE SHAPE (RNNs ONLY)
        return self.get_initial_recurrent_state().shape
    