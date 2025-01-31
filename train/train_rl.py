import os
import time
import glob
import argparse
from copy import deepcopy
from pyvirtualdisplay import Display

from pooling.utils.RL import build_ppo_config, save_checkpoint, load_config, save_config, set_seed


def train(algo, config) -> None:
    total_time = 0
    total_steps = 0
    best_min_rew = {policy_id:-1e+10 for policy_id in config["POLICIES_TO_TRAIN"]}
    best_mean_rew = {policy_id:-1e+10 for policy_id in config["POLICIES_TO_TRAIN"]}
    best_dual_rew = {policy_id:{"mean_rew":-1e+10, "min_rew":-1e+10} for policy_id in config["POLICIES_TO_TRAIN"]}
    for i in range(config['LEARNING_PARAMETERS']["NUM_TRAIN_EPISODES"]):
        start = time.time()
        result = algo.train()
        result.pop("config")
        elapsed = time.time() - start
        # pprint(result)

        ## UPDATE STATS
        total_loss = {pid:info["learner_stats"]["total_loss"] for pid,info in result["info"]["learner"].items()}
        mean_rew = result["env_runners"]["policy_reward_mean"]
        min_rew = result["env_runners"]["policy_reward_min"]
        max_rew = result["env_runners"]["policy_reward_max"]
        
        print("+---------------------------------------------------------------------------------------------------+")
        total_time += elapsed
        total_steps += config["LEARNING_PARAMETERS"]["BATCH_SIZE"]
        epoch = f"{i+1}/{config['LEARNING_PARAMETERS']['NUM_TRAIN_EPISODES']}"
        epoch_pad = ' '*len(epoch)
        time_ = f"{elapsed:.2f}"
        time_pad = ' '*(5 - len(time_))
        print(f"| {epoch} • TIME: {time_pad}{time_}s | Total Time: {total_time/3600:.2f}hrs  -  Total Steps: {total_steps:,}  -  EXP {config['ITERATION']+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
        for policy_id in config["POLICIES_TO_TRAIN"]:
            loss = f"{total_loss[policy_id]:.3f}"
            loss_pad = ' '*(6 - len(loss))
            print(f"| {epoch_pad} | loss: {loss_pad}{total_loss[policy_id]:.3f} | min rew: {min_rew[policy_id]:.2f} | mean rew: {mean_rew[policy_id]:.2f} | max rew: {max_rew[policy_id]:.2f} | policy: {policy_id}")

        ## SAVE POLICY WITH HIGHEST MEAN REWARD
        for policy_id in config["POLICIES_TO_TRAIN"]:
            if mean_rew[policy_id] > best_mean_rew[policy_id]:
                best_mean_rew[policy_id] = mean_rew[policy_id]
                save_checkpoint(algo, policy_id, "mean", mean_rew[policy_id], config["CHECKPOINT_PATH"])
            
            if min_rew[policy_id] > best_min_rew[policy_id]:
                best_min_rew[policy_id] = min_rew[policy_id]
                save_checkpoint(algo, policy_id, "min", min_rew[policy_id], config["CHECKPOINT_PATH"])

            if mean_rew[policy_id] >= best_dual_rew[policy_id]["mean_rew"] \
                and min_rew[policy_id] >= best_dual_rew[policy_id]["min_rew"]:
                best_dual_rew[policy_id]["mean_rew"] = mean_rew[policy_id]
                best_dual_rew[policy_id]["min_rew"] = min_rew[policy_id]
                save_checkpoint(algo, policy_id, "dual", mean_rew[policy_id], config["CHECKPOINT_PATH"])
    return



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/simple-tag-1v3v0", help='Path to the experiment directory.')
    args = parser.parse_args()
    
    ## LOAD AND BUILD CONFIG
    base = __file__.split("train")[0]
    config_path = os.path.join(base, args.experiment_path, "config.yml")
    config = load_config(config_path)
    base_config = deepcopy(config)
    num_experiments = config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]
    
    ## USE VIRTUAL DISPLAY FOR RENDERING ON A HEADLESS SERVER
    disp = Display()
    disp.start()

    ## RUN EXPERIMENTS
    for i in range(num_experiments):
        ## SET SEED
        config = deepcopy(base_config)
        config["ITERATION"] = i
        ppo_config = build_ppo_config(config, exp_dir=os.path.dirname(config_path))

        ## CONFIGURE SAVE PATHS
        experiment_name = config["EXPERIMENT_NAME"]
        config_name = os.path.basename(config_path)
        save_dir = os.path.join(ppo_config["logger_config"]["logdir"])
        save_path = os.path.join(save_dir, config_name)
        
        ## SET SEED
        method = os.path.basename(save_dir).split("_")[0]
        seed = len(glob.glob(os.path.join(os.path.dirname(save_dir), f"{method}_{experiment_name}*")))
        base_config["PPO_CONFIG"]["seed"] = seed
        config["PPO_CONFIG"]["seed"] = seed
        ppo_config["seed"] = seed
        ppo_config["model"]["custom_model_config"]["seed"] = seed
        ppo_config = ppo_config.debugging(seed=seed)
        set_seed(seed)
        save_config(save_path, base_config)
        print(f"\nSTARTING EXPERIMENT {method} {experiment_name}: {i+1}/{num_experiments}\n")

        ## TRAIN
        algo = ppo_config.build()
        train(algo, config)
        algo.stop()
        print("\nTRAINING COMPLETED\n")

    disp.stop()
    
