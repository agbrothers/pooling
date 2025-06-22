import torch
import torch.nn as nn
from torch import Tensor

import ray
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
if ray.__version__ == '1.13.0': import gym
else: import gymnasium as gym

from pooling.utils.diagnostics import get_gpu_memory, convert_size


class RLlibWrapper(nn.Module, TorchModelV2):
    '''
    Generic RLlib wrapper for custom models
    
    ''' 
    
    def __init__(
            self, 
            obs_space, 
            action_space, 
            num_outputs:int,
            model_config:dict, 
            name:str, 
            model:nn.Module, 
            pi:nn.Module, 
            vf:nn.Module, 
            use_gpu:bool=False,
            **custom_model_kwargs
        ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.device = torch.device(
            type = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu',
            index=0 if torch.cuda.is_available() else None
        )
        initial_gpu_mem = get_gpu_memory()

        ## PARSE CONFIG
        custom_model_kwargs.update({
            "dim_action": num_outputs, 
            "dim_token": obs_space.shape[-1],
        })
        # custom_model_kwargs.update({"action_size": self.action_size, "obs_size": self.obs_size})

        ## INITIALIZE POLICY AND VALUE NETWORKS
        self.model = model(pi, vf, **custom_model_kwargs)
        self.recurrent = self.model.pi._recurrent or (self.model.vf and self.model.vf._recurrent)

        self.to(self.device)
        if self.device.type != "cpu":
            print(f"{self._parameter_count:_} LEARNABLE PARAMETERS -> {convert_size(get_gpu_memory()-initial_gpu_mem)}")        

        if not self.recurrent: return
        
        ## ADD ViewRequirements TO INCLUDE RECURRENT STATES IN THE INPUT DICT
        models = (self.model.pi, self.model.vf) if self.model.vf else (self.model.pi,)
        for i,model in enumerate(models):
            space = gym.spaces.Box(-1.0, 1.0, shape=model.get_recurrent_state_shape())
            if f"state_in_{i}" not in self.view_requirements:
                self.view_requirements[f"state_in_{i}"] = ViewRequirement(
                    f"state_out_{i}",
                    # shift=f"-{model.mem_len}:-1",
                    shift=-1,
                    used_for_compute_actions=True,
                    space=space,
                )
            if f"state_out_{i}" not in self.view_requirements:
                self.view_requirements[f"state_out_{i}"] = ViewRequirement(
                    space=space, 
                    used_for_training=True,
                )
        return

    @override(TorchModelV2)
    def forward(self, input_dict, state=None, seq_lens=None):
        ## PARSE INPUT DICT
        obs = input_dict["obs"]

        ## UPDATE RECURRENT STATE
        self.set_recurrent_state(input_dict)

        ## INFERENCE MODEL
        action_logits = self.model(obs)
        self._cur_value = self.model.get_value()

        ## BUILD OUTPUT STATE
        output_state = self.get_output_state()
        return action_logits, output_state

    @override(TorchModelV2)
    def value_function(self) -> Tensor:
        ## RLlib looks for _cur_value
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def get_initial_state(self):
        if not self.recurrent: return []
        pi_recurrent_state = [self.model.pi.get_initial_recurrent_state()]
        vf_recurrent_state = [self.model.vf.get_initial_recurrent_state()] if self.model.vf else []
        return pi_recurrent_state + vf_recurrent_state        

    def set_recurrent_state(self, input_dict):
        ## ADD MODEL SPECIFIC IMPLEMENTATION FOR UPDATING THE INITIAL RECURRENT STATE
        if not self.recurrent: return
        raise NotImplementedError
    
    def get_output_state(self):
        ## ADD MODEL SPECIFIC IMPLEMENTATION FOR BUILDING THE OUTPUT RECURRENT STATE
        if not self.recurrent: return []
        raise NotImplementedError

    @property
    def _parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
