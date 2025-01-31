import os
import glob
import yaml
import argparse
import pandas as pd
import numpy as np


if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/simple-centroid-1v3v0", help='Path to the dataset.')
    args = parser.parse_args()
    
    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    experiment_path = os.path.join(base, args.experiment_path)
    experiment_dirs = sorted(glob.glob(os.path.join(experiment_path, "*")))
    with open(os.path.join(experiment_path, "config.yml"), "r") as file:
        config = yaml.safe_load(file)
    n = config["LEARNING_PARAMETERS"]["NUM_TRAIN_EPISODES"]

    ## WRITE RESULTS FOR EACH EXPERIMENT IN THE SET
    data = []
    for experiment in experiment_dirs:

        if not os.path.isdir(experiment):
            continue

        df = pd.read_csv(os.path.join(experiment, "progress.csv"))
        if len(df) < n:
            continue
        if len(df) > n:
            continue

        method = os.path.basename(experiment).split("_")[0]
        data.append([method] + list(df["env_runners/policy_reward_mean/attenuator"].values))

    columns = ["Method"] + list(np.arange(n).astype(str))
    results = pd.DataFrame(data=data, columns=columns)
    results.to_csv(os.path.join(experiment_path, "results.csv"), index=False)
