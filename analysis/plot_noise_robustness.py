import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


"""
## NAIVE CENTROID BASELINE COMPUTED THE MAIN KNN-CENTROID DATASET AS FOLLOWS
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = batch_X.mean(axis=1)
        loss = criterion(predictions, batch_y)
        test_loss += loss.item()

## SAVE THE FINAL MODEL
test_loss = test_loss / len(test_loader)
print(test_loss)
"""
CENTROID_BASELINE = {
    "knn1": 0.09342181354538710000,
    "knn2": 0.07112235764958960000,
    "knn4": 0.05535614676773540000,
    "knn8": 0.04255681988129860000,
    "knn16": 0.03086179232141420000,
    "knn32": 0.01952238504621960000,
    "knn64": 0.00845499598740864000,
    "knn128": 0.00000770717038130218,
}

"""
## NAIVE TARGET BASELINE COMPUTED THE MAIN KNN-CENTROID DATASET AS FOLLOWS
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = batch_X[:,0]
        loss = criterion(predictions, batch_y)
        test_loss += loss.item()

## SAVE THE FINAL MODEL
test_loss = test_loss / len(test_loader)
print(test_loss)
"""
TARGET_BASELINE = {
    "knn1": 0.0583144005165616,
    "knn2": 0.0442782285164541,
    "knn4": 0.0397463439783053,
    "knn8": 0.0413157179562458,
    "knn16": 0.0477641089193856,
    "knn32": 0.0595708504430393,
    "knn64": 0.0796918134524751,
    "knn128": 0.1262742781372210,
}


def standard_error(data):
    sample_size = data.shape[0]
    std = data.std(axis=0)
    return std / np.sqrt(sample_size)

def plot_std_err(
        data:dict, 
        experiment_path:str,
    ):
    """
    Plot the median and interquartile range of a dataset.
    
    Parameters:
    - data (np.ndarray): 2D array where axis 0 corresponds to samples and axis 1 to timesteps.
    - color (str): Color of the median line and IQR shading.
    - label (str): Label for the median line.
    """

    LINE_ALPHA = 0.7
    BASELINE_ALPHA = 0.3
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
    lineweight = 2
    label_size = 10
    max_loss = -1e+6

    experiments = { 
        int(name.split("knn")[-1]):name 
        for name in data["Experiment"].unique() 
        if "knn" in name 
    }
    experiments = dict(sorted(experiments.items()))
    N = max(experiments.keys())

    fig, ax = plt.subplots(figsize=(5.5,4), dpi=500, nrows=1, ncols=1)
    
    signal_loss_mean = { method:[] for method in METHODS }
    for k,name in experiments.items():
            experiment_data = data[data["Experiment"] == name][["Method", "Test Loss"]]
            mean = experiment_data.groupby("Method").mean()
            for method in METHODS:
                signal_loss_mean[method].append(mean.loc[method].item())

    ## PLOT MEDIAN + IQR
    x = np.round(np.array(list(experiments.keys())) / N, 2)

    ## PLOT IQR
    lines = {}
    for method,label in METHODS.items():
        mean = np.array(signal_loss_mean[method])
        color = COLORS[method]
        lines[method] = plt.plot(x, mean, alpha=LINE_ALPHA, lw=lineweight, c=color, label=label)

    ## PLOT BASELINES
    lines["CTD"] = plt.plot(x, list(CENTROID_BASELINE.values()), alpha=BASELINE_ALPHA, linestyle="--", c=tuple(COLORS["AvgPool"]), label="Centroid Baseline") 
    lines["TGT"] = plt.plot(x, list(TARGET_BASELINE.values()), alpha=BASELINE_ALPHA, linestyle="--", c=tuple(COLORS["MaxPool"]), label="Target Baseline") 

    ## GET Y-AXIS BOUNDS
    max_loss = max(TARGET_BASELINE.values())
    
    # ## SET TRAIN LINE
    font = font_manager.FontProperties(
        weight='medium',
        style='normal', 
        size=10,
    )
    ax.tick_params(axis='both', labelsize=9) 
    ax.set_xlabel(f'Signal-to-Noise Ratio', fontsize=label_size, weight=w)
    ax.set_ylabel('Signal Loss', fontsize=label_size, weight=w)    
    ax.set_ylim((0.0, max_loss*1.05))
    ax.legend(frameon=True, loc="upper right", mode="expand", ncol=3, prop=font) ## 
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path,"noise_robustness.png"))
    plt.cla()
    return


if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/final/k_4x", help='Path to the dataset.')
    args = parser.parse_args()

    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    experiment_path = os.path.join(base, args.experiment_path)    
    with open(os.path.join(experiment_path, "results.csv"), "r") as file:
        data = pd.read_csv(file)
        
    plot_std_err(data, experiment_path)
