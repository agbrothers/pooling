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
        exp_prefix:str,
        exp_seed:int,
        output_dir:str,
        plot_type="Loss"
    ):
    """
    Plot accuracy and loss validation curves from a classification experiment. Choose an 
    individual experiment by setting exp_seed. If exp_seed is None, it will plot the 
    results from all seeds in the experiment directory. 
    
    Parameters:
    - data (np.ndarray): 2D array where axis 0 corresponds to samples and axis 1 to timesteps.
    - color (str): Color of the median line and IQR shading.
    - label (str): Label for the median line.
    """

    LINE_ALPHA = 0.6
    COLORS = {
        "ADA-FOCAL": [ 10/255, 209/255,   0/255], #[ 233/255, 110/255, 8/255], #0bd100  -  orange
        "AVG": [102/255, 158/255, 255/255], #669eff  -  blue
        "CLS":   [255/255, 162/255,   0/255], #ffa200  -  yellow
        "MAX": [255/255,  62/255,  62/255], #ff3e3e  -  red
        "ADA-CORNER": [ 174/255, 130/255, 250/255], #0bd100  -  purple
        "ADA-CENTROID": [ 0/255, 0/255, 255/255], #[233/255, 110/255, 8/255], #[ 141/255, 82/255, 49/255], #0bd100  -  brown
        "ADA-FOCAL-CROSS": [ 0/255, 0/255, 255/255], #0bd100  -  grey
        # "ADA-FOCAL2": [ 0/255, 0/255, 255/255], #0bd100  -  blue
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
    labels = []
    max_y = 0
    for name, df in dfs.items():
        ## SKIP EMPTY DF
        if len(df) == 0: 
            continue

        ## LABEL FOR LEGEND
        label = color = name.split("_")[0]
        if label in labels:
            label = None
        labels.append(label)

        ## PLOT CURVE
        lines[name+"_val"] = plt.plot(df.index[:-1], df[f"Val {plot_type}"][:-1].values.astype(float), alpha=LINE_ALPHA, lw=line_weight, ls="-", c=tuple(COLORS[color]), label=label)
        # lines[name+"_train"] = plt.plot(df.index[:-1], df[f"Train {plot_type}"][:-1].values.astype(float), alpha=LINE_ALPHA, lw=line_weight, ls="-", c=tuple(COLORS[color]), label=label)
        max_y = max(max_y, max(df[f"Val {plot_type}"][:-1].values.astype(float)))

    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        weight='medium',
        style='normal', 
        size=8,
    )
    if plot_type == "Acc":
        ax.set_ylim((2*max_y/3, max_y + 0.02))
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlabel(f'Epochs', fontsize=label_size, weight=w)
    ax.set_ylabel(Y_LABELS[plot_type], fontsize=label_size, weight=w)    
    loc="upper right" if plot_type == "Loss" else "lower right"
    ax.legend(frameon=True, loc=loc, mode="expand", ncol=3, prop=font) ## 
    # ax.legend(frameon=True, loc="upper right", prop=font) 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_prefix}{exp_seed or 'ALL'}-{plot_type.upper()}.png"))
    plt.cla()
    return



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', '--exp_path', default="experiments/cifar100", type=str, help='Path to the root directory of an experiment.')
    parser.add_argument('-g', '--exp_prefix', default="CIFAR-", type=str, help='COMMON PREFIX IN EACH EXPERIMENT_NAME.')
    parser.add_argument('-n', '--exp_seed', default=None, type=int, help='SEED OF THE EXPERIMENT TO LOOK FOR.')
    args = parser.parse_args()

    ## CONFIGURE PATHS
    base = __file__.split("analysis")[0]
    seed = args.exp_seed or ''
    output_dir = os.path.join(base, args.exp_path) 
    exp_path = os.path.join(output_dir, "*", args.exp_prefix+"*"+f"_{seed}{'_'*bool(seed)}*") 
    exp_paths = glob.glob(exp_path)

    get_name = lambda path: os.path.basename(path).split("_")[0].replace(args.exp_prefix, "") + "_" + os.path.basename(path).split("_")[1]
    dfs = {get_name(path):pd.read_csv(os.path.join(path, "history.csv")) for path in exp_paths}
    dfs = dict(sorted(dfs.items()))
    plot(dfs, args.exp_prefix, args.exp_seed, output_dir, plot_type="Loss")
    plot(dfs, args.exp_prefix, args.exp_seed, output_dir, plot_type="Acc")
