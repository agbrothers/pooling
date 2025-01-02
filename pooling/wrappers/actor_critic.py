import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pooling.nn.mlp import MLP
from pooling.nn.embedding import EmbeddingConcatLayer


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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
            share_layers=True, 
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
        self._idx_mask = idx_mask
        self._share_layers = share_layers

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
                dim_hidden=self.pi._dim_hidden, 
                dim_output=self._dim_action,
                num_layers=output_head_pi_layers,
            )
        ## MAPPING OF VALUE FUNCTION OUTPUT TO VALUE SCALAR 
        if self.has_output_head_vf:
            dim_hidden = self.pi._dim_hidden if not self.vf else self.vf._dim_hidden
            self.value_head = MLP(
                dim_input=dim_hidden, 
                dim_hidden=dim_hidden, 
                dim_output=1,
                num_layers=output_head_vf_layers,
            )
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

        ## SCALE VALUE PREDICTION
        # self.value = symexp(self.value)

        return action_logits

    def value_function(self) -> Tensor:
        ## RLlib looks for _cur_value
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def get_value(self):
        return self.value


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
            share_layers=True, 
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
        self._idx_mask = idx_mask
        self._share_layers = share_layers

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
                dim_hidden=self.pi._dim_hidden, 
                dim_output=self._dim_action,
                num_layers=output_head_pi_layers,
            )
        ## MAPPING OF VALUE FUNCTION OUTPUT TO VALUE SCALAR 
        if self.has_output_head_vf:
            dim_hidden = self.pi._dim_hidden if not self.vf else self.vf._dim_hidden
            self.value_head = MLP(
                dim_input=dim_hidden, 
                dim_hidden=dim_hidden, 
                dim_output=1,
                num_layers=output_head_vf_layers,
            )
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

        ## SCALE VALUE PREDICTION
        # self.value = symexp(self.value)

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
