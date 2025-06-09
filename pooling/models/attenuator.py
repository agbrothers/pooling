import torch
import torch.nn as nn
from torch import Tensor

import pooling.nn.pooling as pooling
from pooling.nn.transformer import Transformer


class Attenuator(nn.Module):

    METHODS = {
        "MaxPool": pooling.MaxPool,
        "AvgPool": pooling.AvgPool,
        "AdaPool": pooling.AdaPool,
        "ClsToken": pooling.ClsToken,
    }

    def __init__(
            self, 
            dim_hidden,
            dim_input=None,
            dim_output=None,
            num_emb=2,            
            pos_emb=False,
            pooling_method="AdaPool",
            seed=None,
            **kwargs,
        ) -> None: 
        super().__init__()

        ## SET MODEL PROPERTIES
        self._dim_hidden = dim_hidden
        self._dim_input = dim_input
        self._dim_output = dim_output
        self._pooling_method = pooling_method
        self._recurrent = False

        ## INITIALIZE ENCODER
        self.transformer = Transformer(
            dim_hidden,
            **kwargs,
        )
        self.proj_in = None
        if dim_input:
            self.proj_in = nn.Linear(dim_input, dim_hidden, bias=bias_ff)

        self.proj_out = None
        if dim_output:
            self.proj_out = nn.Linear(dim_hidden, dim_output, bias=bias_ff)

        ## EMBEDDING TO DIFFERENTIATE QUERY FROM OTHER INPUTS
        self.query_emb = None
        if pos_emb:
            self.query_emb = nn.Embedding(num_embeddings=num_emb, embedding_dim=dim_hidden)

        self.pool = self.METHODS[pooling_method](dim_hidden=dim_hidden, **kwargs)

        if seed:
            torch.manual_seed(seed)
            self.apply(self.initialize)
        return


    def forward(self, x, mask=None) -> Tensor:
        ## [Optional] PROJECT INPUT TO HIDDEN DIMENSION
        if self.proj_in:
            x = self.proj_in(x)
        
        ## [Optional] SOURCE CLS TOKEN
        if hasattr(self.pool, "cls_token"):
            x = torch.cat((self.pool.get_cls_token(x), x), dim=1)
        
        ## [Optional] ADD QUERY EMBEDDING
        if self.query_emb:
            idx = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
            idx[:, 0] = 0
            x = x + self.query_emb(idx)

        ## STEP TRANSFORMER ENCODER
        x = self.transformer(x, mask)

        ## APPLY POOLING
        pool = self.pool(x, mask)

        ## [Optional] PROJECT HIDDEN TO OUTPUT DIMENSION
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

    ## HELPER FUNCTIONS FOR RLLIB?
    ##

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
    

if __name__ == "__main__":

    ## CHECK THAT BASE TRANSFORMERS HAVE IDENTICAL PARAMETERS
    ## DESPITE CONFIGURATION WITH DIFFERENT POOLING LAYERS
    ## NOTE: Want to prevent any undue influence from the  
    ## lottery ticket hypothesis. 

    SEED = 0
    dim_hidden = 16
    dim_ff = 64
    num_layers = 12
    num_heads = 8
    dropout_w = 0.
    dropout_e = 0.
    dropout_ff = 0.1
    bias_attn = False
    bias_ff = True

    torch.manual_seed(SEED)
    a = Attenuator(
        dim_hidden,
        dim_ff,
        num_layers,
        num_heads,
        dropout_w,
        dropout_e,
        dropout_ff,
        bias_attn,
        bias_ff,
        pooling_method="AdaPool", 
        seed=SEED,
    )

    torch.manual_seed(SEED)
    b = Attenuator(
        dim_hidden,
        dim_ff,
        num_layers,
        num_heads,
        dropout_w,
        dropout_e,
        dropout_ff,
        bias_attn,
        bias_ff,
        pooling_method="AvgPool",
        seed=SEED,
    )

    for i in range(num_layers):
        assert torch.all(
            a.transformer.encoder.layers[i].attn.Q.weight ==  
            b.transformer.encoder.layers[i].attn.Q.weight  
        )
        assert torch.all(
            a.transformer.encoder.layers[i].attn.KV.weight ==  
            b.transformer.encoder.layers[i].attn.KV.weight  
        )
        assert torch.all(
            a.transformer.encoder.layers[i].attn.out.weight ==  
            b.transformer.encoder.layers[i].attn.out.weight  
        )
        assert torch.all(
            a.transformer.encoder.layers[i].ff.l_in.weight ==  
            b.transformer.encoder.layers[i].ff.l_in.weight  
        )
        assert torch.all(
            a.transformer.encoder.layers[i].ff.l_out.weight ==  
            b.transformer.encoder.layers[i].ff.l_out.weight  
        )

    pass
