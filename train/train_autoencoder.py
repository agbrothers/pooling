import os
import yaml
import argparse
import numpy as np

import torch
from torch.utils.data import TensorDataset
torch.set_float32_matmul_precision('high')

from train.train_supervised import kfold
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
        X = self.data[idx]
        return X, X  # Input and target are the same

    @property
    def tensors(self):
        return (self.data, self.data)


def load_dataset(experiment_path, dim_vectors):
    # Load data
    data_path = os.path.dirname(experiment_path).replace("experiments", "data")
    X = np.load(os.path.join(data_path, f"X{dim_vectors}.npy"))   # SHAPE: [batch, num_vectors, dim_vectors]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)

    # Create a full dataset
    return AutoencoderDataset(X)


if __name__ == "__main__":

    ## PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/autoencoder", help='Path to the dataset.')
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
