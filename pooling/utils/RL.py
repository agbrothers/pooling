# import re
import os
import time
import yaml
import glob
import torch
import shutil
import random
import numpy as np

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger

from pooling import POLICY_REGISTER, __path__
# from pooling.utils.custom_metrics import CustomMetricCallbacks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSET SEED {seed}\n")
    return

def load_config(path:str) -> dict:
    with open(path, "r") as file:
        # test= yaml.safe_load(file)
        return yaml.safe_load(file)
    
def save_config(path:str, config:dict):
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)    
    return

def save_checkpoint(algo, policy_id:str, checkpoint_type:str, reward:str, checkpoint_dir:str):
    ## OVERWRITE PREVIOUS BEST CHECKPOINT 
    previous_checkpoint = glob.glob(os.path.join(checkpoint_dir, policy_id+f"_{checkpoint_type}_"+"*.pt"))
    if len(previous_checkpoint) > 0:
        os.remove(previous_checkpoint[0])

    ## SAVE NEW BEST CHECKPOINT
    filepath = os.path.join(checkpoint_dir, f"{policy_id}_{checkpoint_type}_{int(reward)}.pt")
    policy = algo.get_policy(policy_id)
    torch.save(policy.model.state_dict(), filepath)
    return

def load_checkpoint(algo, policy_id, checkpoint_dir):
    ## LOAD PREVIOUS BEST CHECKPOINT
    # previous_checkpoint = glob.glob(os.path.join(checkpoint_dir, policy_id+"_mean_"+"*.pt"))[0]
    previous_checkpoint = glob.glob(os.path.join(checkpoint_dir, policy_id+"_dual_"+"*.pt"))[0]
    policy = algo.get_policy(policy_id)
    policy.model.load_state_dict(torch.load(previous_checkpoint, weights_only=True))
    policy.model.eval()
    print(f"\nSuccessfully loaded {previous_checkpoint}\n")
    return

def configure_logging(config:dict, exp_dir:str=None, log_dir:str=None, eval:bool=False) -> dict:
    ## SET UP SAVE DIRECTORY
    if log_dir is None:
        method = config["MODEL_CONFIG"]["custom_model_config"].get("pooling_method", config["MODEL_CONFIG"]["custom_model"])
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        experiment_name = f"_{config['EXPERIMENT_NAME']}" if config['EXPERIMENT_NAME'] else ""
        log_dir = os.path.join(exp_dir, f"{method}{experiment_name}_{timestamp}")

    ## MAKE LOGDIR
    if eval:
        log_dir = os.path.join(log_dir, "eval")
        os.makedirs(log_dir, exist_ok=True)
        config["PPO_CONFIG"]["evaluation_config"]["env_config"]["video_dir"] = os.path.join(log_dir, f'videos')
        config["PPO_CONFIG"]["evaluation_config"]["env_config"]["log_rew"] = True
        config["PPO_CONFIG"]["evaluation_config"]["env_config"]["log_path"] = log_dir
        config["PPO_CONFIG"]["evaluation_config"]["env_config"]["log_agents"] = [
            agent_id for agent_id, policy_id in config["AGENT_TO_POLICY"].items() 
            if policy_id in config["POLICIES_TO_LOG"]
        ]
    else:
        config["CHECKPOINT_PATH"] = os.path.join(log_dir, "checkpoints")
        os.makedirs(config["CHECKPOINT_PATH"], exist_ok=True)

    ## SET LOGDIR
    config["PPO_CONFIG"]["env_config"]["video_dir"] = os.path.join(log_dir, "videos")
    config["PPO_CONFIG"]["evaluation_config"]["env_config"]["video_dir"] = os.path.join(log_dir, "videos")
    config["PPO_CONFIG"]["logger_config"] = {"type": UnifiedLogger, "logdir": log_dir, "log_level":"ERROR"}
    return config


def build_ppo_config(config:dict, exp_dir:str=None, log_dir:str=None, eval:bool=False) -> PPOConfig:
    ## SINGLE LOCAL ENV RUNNER FOR DEBUGGING
    if config["LEARNING_PARAMETERS"]["DEBUG"]: 
        config["PPO_CONFIG"].update({"num_env_runners": 0})
    
    ## UPDATE MODEL GPU CONFIGURATION
    config["MODEL_CONFIG"]["custom_model_config"]["use_gpu"] = config["LEARNING_PARAMETERS"]["NUM_GPUS"] > 0

    ## CONFIGURE LOGGING DIRECTORY
    config = configure_logging(config, exp_dir, log_dir, eval)

    ## BUILD PPO CONFIG
    ppo_config = {}
    ppo_config.update(config["PPO_CONFIG"])
    ppo_config["model"] = config["MODEL_CONFIG"]

    ## SET UP MULTI-AGENT CONFIG
    ppo_config["multiagent"] = {
        "policies": {
            policy_id: PolicySpec(policy_class=POLICY_REGISTER[policy_type])
            for policy_id, policy_type in config["POLICIES"].items()
        },
        "policy_mapping_fn": lambda agent_id, episode, worker: config["AGENT_TO_POLICY"][agent_id],
        "policies_to_train": config["POLICIES_TO_TRAIN"],  
        "observation_fn": None,
        "count_steps_by": config["LEARNING_PARAMETERS"]["COUNT_STEPS_BY"] #"env_steps", #"agent_steps",
    }
    # if eval:
    #     ppo_config["callbacks"] = CustomMetricCallbacks
    #     ppo_config["evaluation_config"]["callbacks"] = CustomMetricCallbacks

    ## TUNE HYPERPARAMETER SEARCH OPTIONS
    # seed: tune.grid_search([01])
    # seed: tune.randint(0999)
    return PPOConfig.from_dict(ppo_config) 
