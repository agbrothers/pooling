import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def plot_iqr(
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
    IQR_ALPHA = 0.08 #0.12
    EDGECOLOR_ALPHA = 0.35
    FACECOLOR_ALPHA = 0.08

    METHODS = {
        "RelPool": "AdaPool",
        "LrnPool": "ClsToken",
        "AvgPool": "AvgPool",
        "MaxPool": "MaxPool",
    }
    COLORS = {
        "AvgPool": [102/255, 158/255, 255/255], #669eff  -  blue
        "MaxPool": [255/255,  62/255,  62/255], #ff3e3e  -  red
        "RelPool": [ 10/255, 209/255,   0/255], #0bd100  -  green
        "LrnPool": [255/255, 162/255,   0/255], #ffa200  -  orange
    }

    ## PLOT PARAMETERS
    w = "heavy"
    n = data.shape[1] - 1
    trunc = None #420 # None
    label_size = 8 
    title_size = 16
    x_rotation = 0 #315
    train_idx = 0 #2
    max_rew = -1e+6
    min_rew =  1e+6
    if trunc:
        data = data.drop(columns=np.arange(trunc, n).astype(str))

    # fig, ax = plt.subplots(figsize=(12,4), dpi=500, nrows=1, ncols=1)
    fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)
    # x_labels = None
    
    ## PLOT MEDIAN + IQR
    medians = data.groupby("Method").median()
    lower_quartiles = data.groupby("Method").quantile(0.25)
    upper_quartiles = data.groupby("Method").quantile(0.75)

    ## PLOT IQR
    iqr = {}
    for method, label in METHODS.items():
        x = np.arange(medians.shape[1]) * batch_size  # x-axis values based on the timestep axis
        median = medians.loc[method]
        lower_quartile = lower_quartiles.loc[method]
        upper_quartile = upper_quartiles.loc[method]

        ## GET Y-AXIS BOUNDS
        if upper_quartile.max() > max_rew:
            max_rew = upper_quartile.max()
        if lower_quartile.min() < min_rew:
            min_rew = lower_quartile.min()
        ax.set_ylim((min_rew*0.8,max_rew))

        ## SHADE THE IQR
        plt.fill_between(x, lower_quartile, upper_quartile, alpha=IQR_ALPHA, color=tuple(COLORS[method])) 
        rng = upper_quartile.iloc[-1] - lower_quartile.iloc[-1]
        iqr[rng] = (method, x[-1], lower_quartile.iloc[-1], upper_quartile.iloc[-1])

    ## PLOT MEDIAN LINES ON TOP OF IQR
    lines = {}
    for method, label in METHODS.items():
        x = np.arange(medians.shape[1]) * batch_size  # x-axis values based on the timestep axis
        median = medians.loc[method]
        lower_quartile = lower_quartiles.loc[method]
        upper_quartile = upper_quartiles.loc[method]
        lines[method] = plt.plot(x, median, alpha=LINE_ALPHA, c=tuple(COLORS[method]), label=label)

    ## EXTEND IQR INTO THE MARGIN FOR READABILITY
    for i,(rng,pts) in enumerate(sorted(iqr.items(), reverse=True)): 
        method = pts[0]
        color = COLORS[method]
        plt.fill_between(
            (pts[1], pts[1]+10*batch_size*(i+1)), 
            (pts[2],pts[2]),
            (pts[3],pts[3]),
            edgecolor=color + [EDGECOLOR_ALPHA],
            facecolor=color + [FACECOLOR_ALPHA],
            # label=METHODS[method],
        ) 

    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        # family='Helvetica',
        weight='medium',
        style='normal', 
        size=10,
    )
    # hfont = {'fontname':'Helvetica'}
    ax.axvline(x=x[-1], color='k', linestyle='-', linewidth=0.8)
    # ax.annotate("Training Set", (train_idx+0.1, 50), fontsize=8, color="r")

    # ax.set_xticks(x)
    # ax.set_xticklabels(x_labels, rotation=x_rotation, ha="left", fontdict={"fontsize":label_size})
    # ax.set_title(f"Simple Tag: {os.path.basename(experiment_path)}", fontsize=title_size, weight=w)
    ax.set_xlabel(f'Training Steps', fontsize=label_size, weight=w)
    ax.set_ylabel('Mean Episode Reward', fontsize=label_size, weight=w)    

    # ax.legend(frameon=False, ncols=len(lines), loc="upper center", prop=font) ## bbox_to_anchor=(1, 1)
    # ax.legend(frameon=True, loc="upper left", prop=font) ## 
    ax.legend(frameon=True, loc="lower center", prop=font) ## 
    # ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower left") ## 
    ## 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path,"reward_iqr.png"))
    plt.cla()
    pass



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="experiments/simple-tag-collision-2v6v8", help='Path to the dataset.')
    args = parser.parse_args()

    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    experiment_path = os.path.join(base, args.experiment_path)    
    with open(os.path.join(experiment_path, "results.csv"), "r") as file:
        data = pd.read_csv(file)
    with open(os.path.join(experiment_path, "config.yml"), "r") as file:
        config = yaml.safe_load(file)
        batch_size = config["LEARNING_PARAMETERS"]["BATCH_SIZE"] * config["ENV_CONFIG"]["mpe_config"]["num_adversaries"]
        
    plot_iqr(data, experiment_path, batch_size)
