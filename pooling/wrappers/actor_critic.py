import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pooling.nn.mlp import MLP
from pooling.nn.embedding import EmbeddingConcatLayer


class ActorCritic(nn.Module):
    '''
    GENERIC ACTOR CRITIC WRAPPER FOR RL MODELS 
    
    '''    

    def __init__(
            self, 
            pi, 
            vf, 
            dim_input, 
            dim_action, 
            idx_mask=None, 
            input_layer_pi=True,
            input_layer_vf=True,
            output_head_pi=True,
            output_head_vf=True,
            output_head_pi_layers=2,
            output_head_vf_layers=2,
            output_head_pi_dim=None,
            output_head_vf_dim=None,
            continuous_actions=False, 
            share_layers=True, 
            seed=None,
            **kwargs
        ):
        super().__init__()

        ## INITIALIZE POLICY AND VALUE NETWORKS
        kwargs["dim_input"] = dim_input
        self.init_networks(pi, vf, share_layers, **kwargs)
        self.recurrent = self.pi._recurrent or (self.vf and self.vf._recurrent)

        ## INPUT PROJECTION ACTIVATION FUNCTION
        self.activation = F.gelu

        ## PARSE CONFIG
        self._dim_input = dim_input
        self._dim_action = dim_action
        self._share_layers = share_layers
        self._continuous_actions = continuous_actions
        self._idx_log_std = dim_action // 2
        self._idx_mask = idx_mask

        ## INPUT PROJECTION LAYERS AND OUTPUT HEADS
        self.has_input_layer_pi = input_layer_pi
        self.has_input_layer_vf = input_layer_vf and self.vf
        self.has_output_head_pi = output_head_pi
        self.has_output_head_vf = output_head_vf
        self.input_layer_pi = None
        self.input_layer_vf = None
        self.action_head    = None
        self.value_head     = None
        self.value          = None

        ## PROJECTION OF POLICY INPUT TO HIDDEN DIMENSION
        if self.has_input_layer_pi:
            self.input_layer_pi = nn.Linear(self._dim_input, self.pi._dim_hidden)

        ## PROJECTION OF VALUE FUNCTION INPUT TO HIDDEN DIMENSION
        if self.has_input_layer_vf and not share_layers:
            self.input_layer_vf = nn.Linear(self._dim_input, self.vf._dim_hidden)

        ## MAPPING OF POLICY OUTPUT TO ACTION DIMENSION
        if self.has_output_head_pi:
            self.action_head = MLP(
                dim_input=self.pi._dim_hidden, 
                dim_hidden=output_head_pi_dim or self.pi._dim_hidden, 
                dim_output=self._dim_action,
                num_layers=output_head_pi_layers,
            )
        ## MAPPING OF VALUE FUNCTION OUTPUT TO VALUE SCALAR 
        if self.has_output_head_vf:
            dim_hidden = self.pi._dim_hidden if not self.vf else self.vf._dim_hidden
            self.value_head = MLP(
                dim_input=dim_hidden, 
                dim_hidden=output_head_vf_dim or dim_hidden, 
                dim_output=1,
                num_layers=output_head_vf_layers,
            )
        ## INITIALIZE WEIGHTS
        if seed:
            torch.manual_seed(seed)
            self.apply(self.initialize)            
        return

    def init_networks(self, pi, vf, share_layers, **kwargs):
        ## POLICY
        self.pi = pi(**kwargs)
        ## VALUE FUNCTION        
        self.vf = vf(**kwargs) if not share_layers else None 

    def input(self, x):
        input_pi = self.activation(self.input_layer_pi(x)) if self.has_input_layer_pi else x
        input_vf = self.activation(self.input_layer_pi(x)) if self.has_input_layer_vf else x.clone()
        return input_pi, input_vf

    def output(self, output_pi, output_vf):
        if self.has_output_head_pi: output_pi = self.action_head(output_pi)
        if self.has_output_head_vf: output_vf = self.value_head(output_vf).flatten()
        assert len(output_vf.shape) == 1
        assert output_pi.shape[-1] == self._dim_action
        return output_pi, output_vf

    def forward(self, x, mask=None):
        
        ## ADD SEQUENCE DIM IF INPUT IS A SINGLE TOKEN
        if len(x.shape) == 3: 
            b, n, d = x.shape
            x = x.reshape(b, -1)            

        ## PROJECT INPUT TO HIDDEN DIMENSIONS
        input_pi, input_vf = self.input(x)

        ## PROJECT OUTPUT EMBEDDINGS TO OUTPUT DIMENSIONS
        embd_pi = self.pi(input_pi)
        embd_vf = self.vf(input_vf) if self.vf else embd_pi.clone()
        action_logits, self.value = self.output(embd_pi, embd_vf)

        ## CLAMP CONTINUOUS LOG STD PREDICTIONS TO PREVENT OVERFLOW DURING SAMPLING
        if self._continuous_actions:
            action_logits[..., self._idx_log_std:] = torch.clamp_(action_logits[..., self._idx_log_std:], min=-1.0, max=1.0) 
        return action_logits

    def value_function(self) -> Tensor:
        ## RLlib looks for _cur_value
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def get_value(self):
        return self.value

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



class TokenActorCritic(nn.Module):
    '''
    GENERIC ACTOR CRITIC WRAPPER FOR RL MODELS WITH TOKENIZED INPUTS
    
    '''    

    def __init__(
            self, 
            pi, 
            vf, 
            dim_token, 
            dim_action, 
            dim_embd=32, 
            num_embd=16, 
            idx_embd=None, 
            idx_mask=None, 
            input_layer_pi=True,
            input_layer_vf=True,
            output_head_pi=True,
            output_head_vf=True,
            output_head_pi_layers=2,
            output_head_vf_layers=2,
            output_head_pi_dim=None,
            output_head_vf_dim=None,     
            continuous_actions=False,        
            share_layers=True, 
            weight_scale=0.2,
            seed=None,
            **kwargs
        ):
        super().__init__()

        ## INITIALIZE POLICY AND VALUE NETWORKS
        self.init_networks(pi, vf, share_layers, **kwargs)
        self.embd = EmbeddingConcatLayer(dim_token, dim_embd, num_embd, idx_embd)
        self.recurrent = self.pi._recurrent or (self.vf and self.vf._recurrent)

        ## INPUT PROJECTION ACTIVATION FUNCTION
        self.activation = F.gelu
        # self.activation = nn.GELU() 

        ## PARSE CONFIG
        self._dim_token = dim_token
        self._dim_action = dim_action
        self._dim_input = self.embd._dim_input
        self._share_layers = share_layers
        self._weight_scale = weight_scale
        self._continuous_actions = continuous_actions
        self._idx_log_std = dim_action // 2
        self._idx_mask = idx_mask

        ## INPUT PROJECTION LAYERS AND OUTPUT HEADS
        self.has_input_layer_pi = input_layer_pi
        self.has_input_layer_vf = input_layer_vf and self.vf
        self.has_output_head_pi = output_head_pi
        self.has_output_head_vf = output_head_vf
        self.input_layer_pi = None
        self.input_layer_vf = None
        self.action_head    = None
        self.value_head     = None
        self.value          = None

        ## PROJECTION OF POLICY INPUT TO HIDDEN DIMENSION
        if self.has_input_layer_pi:
            self.input_layer_pi = nn.Linear(self._dim_input, self.pi._dim_hidden)

        ## PROJECTION OF VALUE FUNCTION INPUT TO HIDDEN DIMENSION
        if self.has_input_layer_vf and not share_layers:
            self.input_layer_vf = nn.Linear(self._dim_input, self.vf._dim_hidden)

        ## MAPPING OF POLICY OUTPUT TO ACTION DIMENSION
        if self.has_output_head_pi:
            self.action_head = MLP(
                dim_input=self.pi._dim_hidden, 
                dim_hidden=output_head_pi_dim or self.pi._dim_hidden, 
                dim_output=self._dim_action,
                num_layers=output_head_pi_layers,
            )
        ## MAPPING OF VALUE FUNCTION OUTPUT TO VALUE SCALAR 
        if self.has_output_head_vf:
            dim_hidden = self.pi._dim_hidden if not self.vf else self.vf._dim_hidden
            self.value_head = MLP(
                dim_input=dim_hidden, 
                dim_hidden=output_head_vf_dim or dim_hidden, 
                dim_output=1,
                num_layers=output_head_vf_layers,
            )
        ## INITIALIZE WEIGHTS
        if seed:
            torch.manual_seed(seed)
            self.apply(self.initialize) 
        return

    def init_networks(self, pi, vf, share_layers, **kwargs):
        ## POLICY
        self.pi = pi(**kwargs)
        ## VALUE FUNCTION        
        self.vf = vf(**kwargs) if not share_layers else None 

    def input(self, x):
        input_pi = self.activation(self.input_layer_pi(x)) if self.has_input_layer_pi else x
        input_vf = self.activation(self.input_layer_pi(x)) if self.has_input_layer_vf else x.clone()
        return input_pi, input_vf

    def output(self, output_pi, output_vf):
        if self.has_output_head_pi: output_pi = self.action_head(output_pi)
        if self.has_output_head_vf: output_vf = self.value_head(output_vf).flatten()
        assert len(output_vf.shape) == 1
        assert output_pi.shape[-1] == self._dim_action
        return output_pi, output_vf

    def forward(self, tokens, mask=None):
        
        ## ADD SEQUENCE DIM IF INPUT IS A SINGLE TOKEN
        if len(tokens.shape) == 2: tokens = tokens.unsqueeze(1) 

        ## GET MASK FROM OBS
        mask = self.get_mask(tokens, mask)

        ## PROJECT INPUT TO HIDDEN DIMENSIONS
        tokens = self.embd(tokens)
        input_pi, input_vf = self.input(tokens)

        ## PROJECT OUTPUT EMBEDDINGS TO OUTPUT DIMENSIONS
        embd_pi = self.pi(input_pi)
        embd_vf = self.vf(input_vf) if self.vf else embd_pi.clone()
        action_logits, self.value = self.output(embd_pi, embd_vf)

        ## CLAMP CONTINUOUS LOG STD PREDICTIONS TO PREVENT OVERFLOW DURING SAMPLING
        if self._continuous_actions:
            action_logits[..., self._idx_log_std:] = torch.clamp_(action_logits[..., self._idx_log_std:], min=-1.0, max=1.0) 
        return action_logits

    def value_function(self) -> Tensor:
        ## RLlib looks for _cur_value
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def get_mask(self, tokens, mask):
        ## RETURN INPUT MASK
        if self._idx_mask is None or mask is not None:
            return None
        ## RETREIVE MASK FROM TOKENS
        return tokens[..., self._idx_mask]

    def get_value(self):
        return self.value

    def initialize(self, module) -> None:
        """ 
        INITIALIZATION SCHEME AS IN 
        [1] https://arxiv.org/pdf/1502.01852.pdf
        [2] https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L163
        
        """
        ## LINEAR LAYERS
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self._weight_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        ## LAYERNORMS
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        ## EMBEDDING WEIGHTS
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self._weight_scale)
        return



if __name__ == "__main__":
    
    ## CHECK THAT BASE TRANSFORMERS HAVE IDENTICAL PARAMETERS
    ## DESPITE CONFIGURATION WITH DIFFERENT POOLING LAYERS
    ## NOTE: Want to prevent any undue influence from the  
    ## lottery ticket hypothesis. 
        
    from pooling.models.attenuator import Attenuator

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
    a = TokenActorCritic(
        pi=Attenuator,
        vf=Attenuator,
        dim_token=5,
        dim_action=5,
        dim_embd=4,
        num_embd=4,
        share_layers=True,
        dim_hidden=dim_hidden,
        dim_ff=dim_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_w=dropout_w,
        dropout_e=dropout_e,
        dropout_ff=dropout_ff,
        bias_attn=bias_attn,
        bias_ff=bias_ff,
        pooling_method="RelPool", 
        seed=SEED,
    )

    torch.manual_seed(SEED)
    b = TokenActorCritic(
        pi=Attenuator,
        vf=Attenuator,
        dim_token=5,
        dim_action=5,
        dim_embd=4,
        num_embd=4,
        share_layers=True,
        dim_hidden=dim_hidden,
        dim_ff=dim_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_w=dropout_w,
        dropout_e=dropout_e,
        dropout_ff=dropout_ff,
        bias_attn=bias_attn,
        bias_ff=bias_ff,
        pooling_method="AvgPool",
        seed=SEED,
    )

    for i in range(num_layers):
        assert torch.all(
            a.pi.transformer.encoder.layers[i].attn.Q.weight ==  
            b.pi.transformer.encoder.layers[i].attn.Q.weight  
        )
        assert torch.all(
            a.pi.transformer.encoder.layers[i].attn.KV.weight ==  
            b.pi.transformer.encoder.layers[i].attn.KV.weight  
        )
        assert torch.all(
            a.pi.transformer.encoder.layers[i].attn.out.weight ==  
            b.pi.transformer.encoder.layers[i].attn.out.weight  
        )
        assert torch.all(
            a.pi.transformer.encoder.layers[i].ff.l_in.weight ==  
            b.pi.transformer.encoder.layers[i].ff.l_in.weight  
        )
        assert torch.all(
            a.pi.transformer.encoder.layers[i].ff.l_out.weight ==  
            b.pi.transformer.encoder.layers[i].ff.l_out.weight  
        )
    pass
