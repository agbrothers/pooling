from pooling.wrappers.rllib_wrapper import RLlibWrapper
from pooling.wrappers.actor_critic import ActorCritic
from pooling.nn.mlp import MLP


class RLlibMLP(RLlibWrapper):
    '''
    RLlib wrapper for the Attenuator 
    '''    
    def __init__(self, *args, **kwargs):
        kwargs["dim_input"] = args[0].shape[0] * args[0].shape[1]
        if "model" not in kwargs:
            kwargs.update({"model": ActorCritic})
        if "pi" not in kwargs:
            kwargs.update({"pi": MLP})
        if "vf" not in kwargs:
            kwargs.update({"vf": MLP})
        super().__init__(*args, **kwargs)
        return
    