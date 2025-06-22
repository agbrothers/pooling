import os
import yaml
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset

# from train.train_supervised import MODELS, load_dataset
from train.train_autoencoder import MODELS, load_dataset


def load_model(path, config, data):
    ## INITIALIZE MODEL
    X_shape = data.dataset.tensors[0].shape
    y_shape = data.dataset.tensors[1].shape
    config.update({
        "num_vectors": X_shape[1],
        # "dim_hidden": X_shape[2],
        "dim_ff": 4*config["dim_hidden"],
        # "flash": False,
    })
    model_class = MODELS[config["model"]]
    model = model_class(**config)
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## LOAD WEIGHTS
    checkpoint = glob.glob(os.path.join(path, "ckpt_*.pt"))[-1]
    print(checkpoint)
    state_dict = torch.load(checkpoint, weights_only=True, map_location=device)

    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model
    

def plot_preds(path:str, model, dataset):
    """
    Plot the median and interquartile range of a dataset.
    
    Parameters:
    - data (np.ndarray): 2D array where axis 0 corresponds to samples and axis 1 to timesteps.
    - color (str): Color of the median line and IQR shading.
    - label (str): Label for the median line.
    """

    ## PLOT PARAMETERS
    w = "heavy"
    label_size = 8 
    title_size = 16

    COLORS = {
        "X": "#669eff", # "blue",
        "y_hat": "#ff3e3e", # "red",
        "y": "#0bd100", # "green",
        "q": "#ffa200", # "orange",
    }

    ## COMPUTE PREDICTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idxs = dataset.indices
    k = 100
    X = dataset.dataset.tensors[0][idxs][:k].to(device)
    y = dataset.dataset.tensors[1][idxs][:k].to(device)
    del dataset    
    model.to(device)
    y_hat = model(X)
    if len(y.shape) == 2:
        y = y.unsqueeze(1)
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(1)

    d = X.shape[-1]
    k = 3
    lim = 4
    # i = 0
    for i in range(0, d-1, 2):
        data = X[k,:,i:i+2].T.to("cpu").numpy()
        true = y[k,:,i:i+2].T.to("cpu").numpy()
        pred = y_hat[k,:,i:i+2].T.detach().to("cpu").numpy()
        zero = X[k,0:1,i:i+2].T.detach().to("cpu").numpy()

        print(i)

        ## GENERATE FIG
        fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)

        size = 3
        ## AUTOENCODER
        pairs = np.stack((true, pred)).transpose(2,1,0)
        for j,pair in enumerate(pairs):
            error = ax.plot(*pair, linestyle="--", c=COLORS["y_hat"], alpha=0.3, lw=1)
            label = ax.annotate(str(j), pair[:,0]+0.02, c=COLORS["y"], alpha=0.3, fontsize=6)
        # ## AGGREGATOR
        # error = ax.plot(*np.stack((true, pred)).T, linestyle="--", c=COLORS["y_hat"], alpha=0.3, lw=1)
        scatter_data = ax.scatter(*data, s=size, alpha=0.7, color=COLORS["X"], label="Data")
        scatter_true = ax.scatter(*true, s=size, color=COLORS["y"], label="True")
        scatter_pred = ax.scatter(*pred, s=size, color=COLORS["y_hat"], label="Pred")
        # scatter_zero = ax.scatter(*zero, s=size, color=COLORS["q"])
        experiment_name = f"{os.path.basename(os.path.dirname(path))} APPROXIMATION".upper()
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.1)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.1)

        ax.set_title(f"{experiment_name}", fontsize=title_size, weight=w)
        ax.set_xlabel(f'Dim {i}', fontsize=label_size, weight=w)
        ax.set_ylabel(f'Dim {i+1}', fontsize=label_size, weight=w) 
        ax.legend(frameon=True, loc="best") 
        ax.set_xlim((-lim,lim))
        ax.set_ylim((-lim,lim))
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"pred_{k}_{i}.png"))
        plt.cla()
        del fig
    return


if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-p', '--experiment_path', default="experiments/autoencoder/RelPool_2024-12-26_23-57-50", help='Path to the dataset.')
    # parser.add_argument('-p', '--experiment_path', default="experiments/autoencoder/RelPool_2025-01-01_23-40-12", help='Path to the dataset.')
    parser.add_argument('-p', '--experiment_path', default="experiments/autoencoder/RelPool_2025-01-01_12-52-49", help='Path to the dataset.')
    args = parser.parse_args()
    
    ## LOAD CONFIG
    base = __file__.split("analysis")[0]
    config_path = os.path.join(base, args.experiment_path)
    with open(os.path.join(config_path, "config.yml"), "r") as file:
        config = yaml.safe_load(file)

    ## LOAD MODEL AND TEST DATA
    dim_vectors = config["LEARNING_PARAMETERS"]["DIM_INPUT"]
    dataset = load_dataset(os.path.dirname(args.experiment_path.replace("final/","")), dim_vectors=dim_vectors)

    test_ratio = config["LEARNING_PARAMETERS"]["TEST_RATIO"]
    data_shape = dataset.tensors[0].shape
    total_size = data_shape[0]
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    test_idxs = list(range(train_size, total_size))
    test_dataset = Subset(dataset, test_idxs)

    ## PLOT CURVES
    model = load_model(args.experiment_path, config["MODEL_CONFIG"], test_dataset)
    plot_preds(args.experiment_path, model, test_dataset)
