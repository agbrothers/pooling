import os
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
from ray.rllib.algorithms.ppo import PPOTorchPolicy

from pettingzoo.mpe import simple_tag_v3

from pooling.envs.recorder import RecordVideoMultiAgent
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
    "random": RandomPolicy,
    "ppo": PPOTorchPolicy,
    "heuristic": RllibHeuristic,
    "predator": PredatorHeuristic,
    "prey": PreyHeuristic,
}


## REGISTER ENVIRONMENT WITH RAY/RLLIB
def configure_mpe(config):
    ## PARSE CONFIG
    cycles_per_recording = config["cycles_per_recording"]
    del config["cycles_per_recording"]
    
    ## OPTIONAL LOGGING
    video_dir = config.get("video_dir", None)
    if video_dir: del config["video_dir"]
    log_path = config.get("log_path", None)
    if log_path: del config["log_path"]
    log_agents = config.get("log_agents", [])
    if log_agents: del config["log_agents"]
    log_rew = config.get("log_rew", False)
    if log_rew: del config["log_rew"]
    log_obs = config.get("log_obs", False)
    if log_obs: del config["log_obs"]
    log_act = config.get("log_act", False)
    if log_act: del config["log_act"]
    
    ## CREATE ENV
    env_creator = lambda kwargs: simple_tag_v3.env(**kwargs)
    return RecordVideoMultiAgent(
        TokenizedMPE(
            env_creator(config), 
            log_path=log_path,
            log_agents=log_agents,
            log_rew=log_rew,
            log_obs=log_obs,
            log_act=log_act,
        ),
        video_folder=video_dir, 
        video_length=config["max_cycles"],
        episode_trigger=lambda t: t % cycles_per_recording == 0 and t > 0, 
        disable_logger=True,            
    )

register_env("simple_tag", configure_mpe)

