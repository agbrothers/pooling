import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def standard_error(data):
    sample_size = data.shape[0]
    std = data.std(axis=0)
    return std / np.sqrt(sample_size)

def plot_se(
        data:dict, 
        experiment_path:str,
        batch_size:int,
    ):
    """
    Plot the median and interquartile range of a dataset.
    
    Parameters:
    - data (np.ndarray): 2D array where axis 0 corresponds to samples and axis 1 to timesteps.
    - color (str): Color of the median line and IQR shading.
    - label (str): Label for the median line.
    """

    LINE_ALPHA = 0.6
    IQR_ALPHA = 0.12
    METHODS = {
        "AdaPool": "AdaPool",
        "ClsToken": "ClsToken",
        "AvgPool": "AvgPool",
        "MaxPool": "MaxPool",
    }
    COLORS = {
        "AvgPool": [102/255, 158/255, 255/255], #669eff  -  blue
        "MaxPool": [255/255,  62/255,  62/255], #ff3e3e  -  red
        "AdaPool": [ 10/255, 209/255,   0/255], #0bd100  -  green
        "ClsToken": [255/255, 162/255,   0/255], #ffa200  -  orange
    }

    ## PLOT PARAMETERS
    w = "heavy"
    k = 1
    label_size = 8 
    max_rew = -1e+6
    min_rew =  1e+6

    fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)
    
    ## PLOT MEDIAN + IQR
    means = data.groupby("Method").mean()
    se = data.groupby("Method").apply(standard_error)
    x = np.arange(means.shape[1], step=k) * batch_size  # x-axis values based on the timestep axis
    moving_avg_window = 5

    ## PLOT IQR
    error = {}
    error_order = dict(sorted({v:k for k,v in means.max(axis=1).items()}.items(), reverse=True)).values()
    for method in error_order:
        mean = means.loc[method][::k]
        upper_se = mean + se.loc[method][::k]
        lower_se = mean - se.loc[method][::k]
        upper_se = upper_se.rolling(window=moving_avg_window, min_periods=1).mean()
        lower_se = lower_se.rolling(window=moving_avg_window, min_periods=1).mean()

        ## GET Y-AXIS BOUNDS
        if upper_se.max() > max_rew:
            max_rew = upper_se.max()
        if lower_se.min() < min_rew:
            min_rew = lower_se.min()

        ## SHADE THE IQR
        color = COLORS[method]
        plt.fill_between(
            x, 
            lower_se, 
            upper_se, 
            edgecolor=color + [0.0],
            facecolor=color + [IQR_ALPHA], 
        )         
        rng = upper_se.iloc[-1] - lower_se.iloc[-1]
        error[rng] = (method, x[-1], lower_se.iloc[-1], upper_se.iloc[-1])

    ## PLOT MEDIAN LINES ON TOP OF IQR
    lines = {}
    for method, label in METHODS.items():
        mean = means.loc[method][::k]
        lines[method] = plt.plot(x, mean, alpha=LINE_ALPHA, c=tuple(COLORS[method]), label=label)

    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        weight='medium',
        style='normal', 
        size=10,
    )
    ax.set_ylim((min_rew*0.8,max_rew*1.05))
    ax.set_xlabel(f'Training Steps', fontsize=label_size, weight=w)
    ax.set_ylabel('Mean Episode Reward', fontsize=label_size, weight=w)    
    ax.legend(frameon=True, loc="upper left", prop=font) 
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path,"reward_se.png"))
    plt.cla()
    return



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/simple-centroid-1v3v0", help='Path to the dataset.')
    args = parser.parse_args()

    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    experiment_path = os.path.join(base, args.experiment_path)    
    with open(os.path.join(experiment_path, "results.csv"), "r") as file:
        data = pd.read_csv(file)
    with open(os.path.join(experiment_path, "config.yml"), "r") as file:
        config = yaml.safe_load(file)
        num_agents = len([a for a,p in config["AGENT_TO_POLICY"].items() if p in config["POLICIES_TO_TRAIN"]])
        batch_size = config["LEARNING_PARAMETERS"]["BATCH_SIZE"] * num_agents
        
    plot_se(data, experiment_path, batch_size)
