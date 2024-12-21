import math
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from pooling.models.attenuator import Attenuator
from pooling.nn.transformer import TransformerLayer


class Decoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer._dim_hidden)
        self._num_layers = num_layers

    def forward(self, query, context, mask=None) -> Tensor: 
        ## HERE THE CONTEXT IS OUR COMPRESSED/POOLED EMBEDDING [b, 1, d]
        ## QUERY IS THE SET OF POSITIONAL EMBEDDINGS [b, k, d]
        for i, layer in enumerate(self.layers): 
            if i==0:
                x = layer(query, context, mask=mask)
            else:
                x = layer(x, mask=mask)
        ## WE RETURN A SET [b, k, d] APPROXIMATING THE NETWORK INPUT
        return x
        # return self.norm(x)
    

class Autoencoder(nn.Module):

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
            flash=True,
            pooling_norm=False,
            pooling_method="relational_query",
            num_embd=32,
            num_out=None,
            expansion=2,
            **kwargs,
        ) -> None: 
        super().__init__()

        ## SET MODEL PROPERTIES
        self._dim_hidden = dim_hidden
        self._dim_ff = dim_ff
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_embd = num_embd
        self._num_out = num_out or num_embd
        self._dropout_w = dropout_w,
        self._dropout_e = dropout_e,
        self._dropout_ff = dropout_ff,
        self._bias_attn = bias_attn,
        self._bias_ff = bias_ff,
        self._pooling_norm = pooling_norm
        self._pooling_method = pooling_method
        self._flash = flash
        self._recurrent = False
        self._exp = expansion

        ## INITIALIE ENCODER
        self.proj_in = nn.Linear(dim_hidden, dim_hidden*self._exp)
        self.proj_out = nn.Linear(dim_hidden*self._exp, dim_hidden)        
        self.encoder = Attenuator(
            dim_hidden*self._exp,
            dim_ff*self._exp,
            num_layers,
            num_heads,
            dropout_w,
            dropout_e,
            dropout_ff,
            bias_attn,
            bias_ff,
            flash,
            pooling_norm,
            pooling_method, 
            **kwargs,
        )
        decoder_layer = TransformerLayer(
            dim_hidden*self._exp,
            dim_ff*self._exp,
            num_heads,
            dropout_w,
            dropout_e,
            dropout_ff,
            bias_attn,
            bias_ff,
            flash,
            **kwargs
        )
        self.decoder = Decoder(
            decoder_layer,
            num_layers,
        )
        self.pos_emb = nn.Embedding(num_embeddings=num_embd, embedding_dim=dim_hidden*self._exp)

        ## INITIALIZE WEIGHTS
        self.apply(self.initialize)
        # for name,param in self.named_parameters():
        #     if name.endswith("out.weight"):
        #         torch.nn.init.normal_(param, mean=0.0, std=0.00001)
                # torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        #     if name.endswith("pool.attn.Q.weight"):
        #         torch.nn.init.normal_(param, mean=0.0, std=0.0002)
        #     # if name.endswith("pool.attn.KV.weight"):
        #     #     with torch.no_grad():
        #     #         param[:dim_hidden] = torch.nn.Parameter(-self.pool.attn.Q.weight.clone().detach())
        #         # torch.nn.init.normal_(param[:dim_hidden], mean=0.0, std=0.0002)
        #         # torch.nn.init.eye_(param[:dim_hidden])
        #         # torch.nn.init.eye_(param[:dim_hidden])
        #         # torch.nn.init.eye_(param[dim_hidden:])
        #     elif name.endswith("pool.attn.out.weight"):
        #         torch.nn.init.eye_(param)        
        return


    def forward(self, x, mask=None) -> Tensor:
        ## GET POSITIONAL EMBEDDINGS
        b, k, d = x.shape
        pos = torch.arange(0, k, dtype=torch.long, device=x.device) 
        pos_emb = self.pos_emb(pos) 

        ## COMPRESS INPUT SET: [b, k, d] -> [b, 1, d]
        x = self.proj_in(x) + pos_emb
        z = self.encoder(x, mask)

        ## RECONSTRUCT INPUT SET: [b, 1, d] -> [b, k, d]
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
        x_hat = self.decoder(
            # query=x[:, :self._num_out], 
            query=torch.tile(pos_emb[:self._num_out], (b,1,1)), 
            context=z
        ) 
        return self.proj_out(x_hat)


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
    