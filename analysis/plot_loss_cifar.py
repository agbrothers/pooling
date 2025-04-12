import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def standard_error(data):
    sample_size = data.shape[0]
    std = data.std(axis=0)
    return std / np.sqrt(sample_size)

def plot(
        dfs:dict, 
        exp_name:str,
        exp_num:int,
        output_dir:str,
        plot_type="Loss"
    ):
    """
    Plot the median and interquartile range of a dataset.
    
    Parameters:
    - data (np.ndarray): 2D array where axis 0 corresponds to samples and axis 1 to timesteps.
    - color (str): Color of the median line and IQR shading.
    - label (str): Label for the median line.
    """

    LINE_ALPHA = 0.6
    COLORS = {
        "AVG": [102/255, 158/255, 255/255], #669eff  -  blue
        "CLS":   [255/255, 162/255,   0/255], #ffa200  -  yellow
        "MAX": [255/255,  62/255,  62/255], #ff3e3e  -  red
        "ADA-MIDDLE": [ 10/255, 209/255,   0/255], #0bd100  -  green
        "ADA-CORNER": [ 174/255, 130/255, 250/255], #0bd100  -  purple
        "ADA-CENTROID": [ 141/255, 82/255, 49/255], #0bd100  -  grey
        "ADA-FOCAL": [ 10/255, 209/255,   0/255], #[ 233/255, 110/255, 8/255], #0bd100  -  orange
        "ADA-FOCAL2": [ 0/255, 0/255, 255/255], #0bd100  -  blue
    }
    Y_LABELS = {
        "Loss": "Cross Entropy Loss",
        "Acc": "Top 1 Accuracy",
    }

    ## PLOT PARAMETERS
    w = "heavy"
    line_weight = 1
    label_size = 10

    fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)
    
    lines = {}
    max_y = 0
    for name, df in dfs.items():
        if name == "ADA-MIDDLE": continue
        # lines[name+"_train"] = plt.plot(df.index[:-1], df[f"Train {plot_type}"][:-1].values.astype(float), alpha=LINE_ALPHA, lw=line_weight, ls="-", c=tuple(COLORS[name]), label=name)
        lines[name+"_val"] = plt.plot(df.index[:-1], df[f"Val {plot_type}"][:-1].values.astype(float), alpha=LINE_ALPHA, lw=line_weight, ls="-", c=tuple(COLORS[name]), label=name)
        max_y = max(max_y, max(df[f"Val {plot_type}"][:-1].values.astype(float)))

    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        weight='medium',
        style='normal', 
        size=8,
    )
    if plot_type == "Acc":
        # ax.set_ylim((0.6, 1.0))
        # ax.set_ylim((0.7, 0.9))
        # ax.set_ylim((0.4, 0.6))
        ax.set_ylim((2*max_y/3, max_y + 0.02))
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlabel(f'Epochs', fontsize=label_size, weight=w)
    ax.set_ylabel(Y_LABELS[plot_type], fontsize=label_size, weight=w)    
    loc="upper right" if plot_type == "Loss" else "lower right"
    ax.legend(frameon=True, loc=loc, mode="expand", ncol=3, prop=font) ## 
    # ax.legend(frameon=True, loc="upper right", prop=font) 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}{exp_num}-{plot_type.upper()}.png"))
    plt.cla()
    return



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', '--exp_path', default="experiments/cifar10", type=str, help='Path to the dataset.')
    parser.add_argument('-g', '--exp_name', default="4-CIFAR-", type=str, help='NAME OF THE EXPERIMENT TO LOOK FOR.')
    parser.add_argument('-n', '--exp_num', default=4, type=int, help='NAME OF THE EXPERIMENT TO LOOK FOR.')
    args = parser.parse_args()

    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    output_dir = os.path.join(base, args.exp_path) 
    exp_path = os.path.join(output_dir, "*", args.exp_name+"*"+f"_{args.exp_num}_*") 
    exp_paths = glob.glob(exp_path)

    dfs = {os.path.basename(path).split("_")[0].replace(args.exp_name, ""):pd.read_csv(os.path.join(path, "history.csv")) for path in exp_paths}
    dfs = dict(sorted(dfs.items()))
    plot(dfs, args.exp_name, args.exp_num, output_dir, plot_type="Loss")
    plot(dfs, args.exp_name, args.exp_num, output_dir, plot_type="Acc")
