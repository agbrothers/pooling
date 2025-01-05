from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
from ray.rllib.algorithms.ppo import PPOTorchPolicy

from pettingzoo.mpe import simple_tag_v3

from pooling.envs.recorder import RecordVideoMultiAgent
from pooling.heuristics.random import RandomHeuristic
from pooling.envs.mpe_tokenizer import TokenizedMPE
from pooling.heuristics.heuristic import RllibHeuristic
from pooling.heuristics.predator import PredatorHeuristic
from pooling.heuristics.prey import PreyHeuristic



## REGISTER CUSTOM MODELS WITH RAY/RLLIB
from pooling.wrappers.rllib_attenuator import RLlibAttenuator
ModelCatalog.register_custom_model(
    "Attenuator", RLlibAttenuator,
)
# from pooling.wrappers.rllib_attenuator import RLlibAttenuator
# ModelCatalog.register_custom_model(
#     "Autoencoder", RLlibAttenuator,
# )
from pooling.wrappers.rllib_mlp import RLlibMLP
ModelCatalog.register_custom_model(
    "MLP", RLlibMLP,
)

## REGISTER POLICIES
POLICY_REGISTER = {
    # "random": RandomPolicy,
    "random": RandomHeuristic,
    "ppo": PPOTorchPolicy,
    "heuristic": RllibHeuristic,
    "predator": PredatorHeuristic,
    "prey": PreyHeuristic,
}


## REGISTER ENVIRONMENT WITH RAY/RLLIB
def configure_mpe(config):
    ## CREATE ENV
    env_creator = lambda kwargs: simple_tag_v3.env(**kwargs)
    return RecordVideoMultiAgent(
        TokenizedMPE(
            env_creator(config["mpe_config"]), 
            **config,
        ),
        video_folder=config["video_dir"], 
        video_length=config["mpe_config"]["max_cycles"],
        episode_trigger=lambda t: t % config["episodes_per_recording"] == 0 and t > 0, 
        disable_logger=True,            
    )

register_env("simple_tag", configure_mpe)

