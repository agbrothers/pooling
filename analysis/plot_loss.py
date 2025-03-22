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
        attn_df:pd.DataFrame, 
        gem_df:pd.DataFrame, 
        output_dir:str,
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
        "Train": [102/255, 158/255, 255/255], #669eff  -  blue
        "Val":   [255/255, 162/255,   0/255], #ffa200  -  orange
        "MaxPool": [255/255,  62/255,  62/255], #ff3e3e  -  red
        "AdaPool": [ 10/255, 209/255,   0/255], #0bd100  -  green
    }

    ## PLOT PARAMETERS
    w = "heavy"
    label_size = 8 

    fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)
    
    lines = {}
    lines["attn"] = plt.plot(attn_df.index, attn_df["Train Loss"], alpha=LINE_ALPHA, ls="-", c=tuple(COLORS["Train"]), label="Attn")
    lines["gem"] = plt.plot(gem_df.index, gem_df["Train Loss"], alpha=LINE_ALPHA, ls="--", c=tuple(COLORS["Train"]), label="Gem")
    lines["attn"] = plt.plot(attn_df.index, attn_df["Val Loss"], alpha=LINE_ALPHA, ls="-", c=tuple(COLORS["Val"]), label="Attn")
    lines["gem"] = plt.plot(gem_df.index, gem_df["Val Loss"], alpha=LINE_ALPHA, ls="--", c=tuple(COLORS["Val"]), label="Gem")

    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        weight='medium',
        style='normal', 
        size=10,
    )
    # ax.set_ylim((min_rew*0.8,max_rew*1.05))
    ax.set_xlabel(f'Epochs', fontsize=label_size, weight=w)
    ax.set_ylabel('Cross Entropy Loss', fontsize=label_size, weight=w)    
    ax.legend(frameon=True, loc="upper right", prop=font) 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"loss.png"))
    plt.cla()
    return



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', '--attn_path', default="./experiments/gem-baseline/cnd-attn/exp_2025-03-21_16-51-16/history.csv", help='Path to the dataset.')
    parser.add_argument('-g', '--gem_path', default="./experiments/gem-baseline/cnd-gem/exp_2025-03-21_16-53-35/history.csv", help='Path to the dataset.')
    parser.add_argument('-o', '--output_dir', default="./experiments/gem-baseline", help='Path to the dataset.')
    args = parser.parse_args()

    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    attn_path = os.path.join(base, args.attn_path)    
    gem_path = os.path.join(base, args.gem_path)    
    output_dir = os.path.join(base, args.output_dir)    

    attn_df = pd.read_csv(attn_path)
    gem_df = pd.read_csv(gem_path)

    plot_se(attn_df, gem_df, output_dir)
