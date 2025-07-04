from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOTorchPolicy

from pooling.envs import mpe_tag
from pooling.envs import mpe_centroid
from pooling.envs.boxworld import BoxWorldRllib
from pooling.envs.mpe_tokenizer import TokenizedMPE 
from pooling.envs.recorder import RecordVideoMultiAgent
from pooling.heuristics.random import RandomHeuristic
from pooling.heuristics.heuristic import RllibHeuristic
from pooling.heuristics.predator import PredatorHeuristic
from pooling.heuristics.prey import PreyHeuristic


## REGISTER CUSTOM MODELS WITH RAY/RLLIB
from pooling.wrappers.rllib_attenuator import RLlibAttenuator
ModelCatalog.register_custom_model(
    "Attenuator", RLlibAttenuator,
)
from pooling.wrappers.rllib_mlp import RLlibMLP
ModelCatalog.register_custom_model(
    "MLP", RLlibMLP,
)

## REGISTER POLICIES
POLICY_REGISTER = {
    "random": RandomHeuristic,
    "ppo": PPOTorchPolicy,
    "heuristic": RllibHeuristic,
    "predator": PredatorHeuristic,
    "prey": PreyHeuristic,
}

ENV_REGISTER = {
    "mpe_tag": mpe_tag,
    "mpe_centroid": mpe_centroid,
}


## REGISTER CUSTOM ENVIRONMENTS WITH RAY/RLLIB
def configure_mpe(config):
    ## CREATE ENV
    env_creator = lambda kwargs: ENV_REGISTER[config["env_id"]].env(**kwargs)
    return RecordVideoMultiAgent(
        TokenizedMPE(
            env_creator(config), 
            **config,
        ),
        video_folder=config["video_dir"], 
        video_length=config["max_cycles"],
        episode_trigger=lambda t: t % config["episodes_per_recording"] == 0 and t > 0, 
        disable_logger=True,            
    )

def configure_boxworld(config):
    ## CREATE ENV
    return RecordVideoMultiAgent(
        BoxWorldRllib(**config),
        video_folder=config["video_dir"], 
        video_length=-1, #config["max_steps"],
        episode_trigger=lambda t: t % config["episodes_per_recording"] == 0 and t > 0, 
        disable_logger=True,            
    )

register_env("mpe_tag", configure_mpe)
register_env("mpe_centroid", configure_mpe)
register_env("boxworld", configure_boxworld)
