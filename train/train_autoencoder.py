import os
import time
import yaml
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
torch.set_float32_matmul_precision('high')

from .train_supervised import kfold
from pooling.models.attenuator import Attenuator
from pooling.models.autoencoder import Autoencoder
from pooling.models.aggregation_mlp import AggregationMLP


MODELS = {
    "Attenuator": Attenuator,
    "Autoencoder": Autoencoder,
    "AggregationMLP": AggregationMLP
}

class AutoencoderDataset(TensorDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # Input and target are the same

    @property
    def tensors(self):
        return (self.data, self.data)


def load_dataset(experiment_path, seed):
    # Load data
    method = os.path.basename(experiment_path)
    data_path = os.path.dirname(experiment_path)
    X = np.load(os.path.join(data_path, f"X.npy"))   # SHAPE: [batch, num_vectors, dim_vectors]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)

    # Create a full dataset
    full_dataset = AutoencoderDataset(X)

    # Split into train, validation, and test datasets
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure total size matches

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)  # Ensures deterministic split
    )
    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./datasets/autoencoder", help='Path to the dataset.')
    args = parser.parse_args()

    # paths = [
    #     "./experiments/min",
    #     "./experiments/max",
    #     "./experiments/mean",
    #     "./experiments/knn1",
    #     "./experiments/knn2",
    #     "./experiments/knn4",
    #     "./experiments/knn8",
    #     "./experiments/knn16",
    #     # "./experiments/subset",
    #     # "./experiments/combo",
    # ]
    # base = __file__.split("train")[0]
    # configs = []
    # for path in paths:
    #     with open(os.path.join(base, path, "config.yml"), "r") as file:
    #         configs.append(yaml.safe_load(file))

    # for path,config in list(zip(paths, configs)):
    #     ## TRAIN
    #     method = config["MODEL_CONFIG"].get("pooling_method", "mlp") #.upper()
    #     target = os.path.basename(path).upper()
    #     path = os.path.join(base, path)
    #     for i in range(config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]):
    #         print(f"\nSTARTING EXPERIMENT FOR {method} POOLING APPROXIMATION OF {target}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
    #         kfold(path, config, load_dataset)


    ## LOAD CONFIG
    base = __file__.split("train")[0]
    path = os.path.join(base, args.experiment_path)    
    with open(os.path.join(path, "config.yml"), "r") as file:
        config = yaml.safe_load(file)

    ## TRAIN
    method = config["MODEL_CONFIG"].get("pooling_method", "mlp") #.upper()
    target = os.path.basename(args.experiment_path).upper()
    for i in range(config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]):
        print(f"\nSTARTING EXPERIMENT FOR {method} APPROXIMATION OF {target}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
        kfold(path, config, load_dataset)
        # train(args.experiment_path, config, load_dataset)
