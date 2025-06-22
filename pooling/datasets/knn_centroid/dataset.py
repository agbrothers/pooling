import os
import torch
import argparse
import numpy as np
from tqdm import trange
from torch.utils.data import TensorDataset


def softmax(x, axis=None):
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)

def knn(k):
    """
    Return the nearest neighbors to the vector at index 0. 
    
    """
    def nearest_neighbors(x, axis):
        dists = np.linalg.norm(x - x[:, 0:1], axis=-1)
        idx = np.argsort(dists, axis=axis)
        nearest_k = np.take_along_axis(x, idx[..., None], axis=axis)[:, 1:k+1]
        return np.mean(nearest_k, axis=axis)
    
    return nearest_neighbors


def kfn(k):
    """
    Return the furthest neighbors from the vector at index 0. 

    """
    def furthest_neighbors(x, axis):
        dists = np.linalg.norm(x - x[:, 0:1], axis=-1)
        idx = np.argsort(dists, axis=axis)
        furthest_k = np.take_along_axis(x, idx[..., None], axis=axis)[:, -k:]    
        return np.mean(furthest_k, axis=axis)    

    return furthest_neighbors
    

def generate_X(
        num_samples:int, 
        num_vectors:int, 
        dim_vectors:int, 
    ):

    num_distributions = 3
    norm = np.sqrt(dim_vectors)
    dataset_shape = (num_samples, num_vectors, dim_vectors)
    sample_shape = (num_vectors, dim_vectors//num_distributions+1)
    samples = []
    ## CONSTRUCT SAMPLES WITH FEATURES DRAWN FROM A MIX OF DISTRIBUTIONS
    for _ in trange(num_samples):

        ## NORMAL DISTRIBUTION WITH RANDOMLY SAMPLED MEAN/STD
        mean = np.random.uniform(low=-3, high=3, size=sample_shape[1])
        std = np.random.uniform(low=1, high=3, size=sample_shape[1])
        normal = np.random.normal(
            loc=mean, 
            scale=std, 
            size=sample_shape, 
        )
        ## UNIFORM DISTRIBUTION WITH RANDOMLY SAMPLED BOUNDS
        low = np.random.uniform(low=-3, high=3, size=sample_shape[1])
        high = low + np.random.uniform(low=0.2, high=3, size=sample_shape[1])
        uniform = np.random.uniform(
            low=low, 
            high=high, 
            size=sample_shape
        )
        ## EXPONENTIAL DISTRIBUTION WITH RANDOM SCALE, DIRECTION, & SHIFT
        scale = np.random.uniform(low=0.1, high=2, size=sample_shape[1])
        sign = np.random.choice([-1,1], size=sample_shape[1])
        shift = -sign * np.random.uniform(low=0, high=3, size=sample_shape[1])
        exponential = np.random.exponential(
            scale=scale, 
            size=sample_shape
        ) * sign + shift
        
        ## STACK FEATURE DISTRIBUTIONS
        sample = np.hstack((
            normal,
            uniform,
            exponential
        )).T / norm

        ## SHUFFLE ORDER OF FEATURE DISTRIBUTIONS
        np.random.shuffle(sample)

        ## TRANSPOSE TO GET [num_vectors, dim_vectors]
        samples.append(sample[:dim_vectors].T)

    return np.stack(samples) 


def generate_y(
        X:np.ndarray, 
        aggr_method:str
    ):
    METHODS = {
        "avg": np.mean,
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "knn1": knn(1),
        "knn2": knn(2),
        "knn3": knn(3),
        "knn4": knn(4),
        "knn8": knn(8),
        "knn16": knn(16),
        "knn32": knn(32),
        "knn64": knn(64),
        "knn128": knn(128),
    }
    return METHODS[aggr_method](X, axis=1)


def load_knn_centroid(experiment_path, config):
    ## PARSE CONFIG
    cardinality = config["LEARNING_PARAMETERS"]["INPUT_CARDINALITY"]
    dim_vectors = config["LEARNING_PARAMETERS"]["INPUT_DIM"]
    num_neighbors = config["LEARNING_PARAMETERS"]["NUM_NEIGHBORS"]
    test_ratio = config["LEARNING_PARAMETERS"]["TEST_RATIO"]

    ## LOAD DATA
    base_path = experiment_path.split("experiments")[0]
    data_path = os.path.join(base_path, "pooling", "datasets", "knn_centroid")
    X = np.load(os.path.join(data_path, f"X-N{cardinality}-d{dim_vectors}.npy"))                 # SHAPE: [batch, num_vectors, dim_vectors]
    y = np.load(os.path.join(data_path, f"y-N{cardinality}-d{dim_vectors}-KNN{num_neighbors}.npy")) # SHAPE: [batch, aggregate_vector]

    # Convert to torch tensors
    split = len(X) - int(len(X) * test_ratio)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create a full dataset
    trainset = TensorDataset(X[:split], y[:split])
    testset  = TensorDataset(X[split:], y[split:])
    return trainset, testset



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--method', default="knn128", type=str, help='Aggregation method for the dataset.')
    parser.add_argument('-n', '--num_samples', default=1_000_000, type=int, help='Number of samples in the dataset.')
    parser.add_argument('-k', '--num_vectors', default=128, type=int, help='Number of vectors to aggregate per sample.')
    parser.add_argument('-d', '--dim_vectors', default=16, type=int, help='Dimensionality of the vectors per sample.')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Seed for dataset generation.')
    args = parser.parse_args()
    np.random.seed(args.seed)

    X_path = os.path.join(os.path.dirname(__file__), f"X-N{args.num_vectors}-d{args.dim_vectors}.npy")
    y_path = os.path.join(os.path.dirname(__file__), f"y-N{args.num_vectors}-d{args.dim_vectors}-{args.method.upper()}.npy")
    print(X_path)
    print(y_path)
    ## USE SAME X TRAINING DATA FOR EACH AGGREGATION METHOD TO SAVE SPACE
    if os.path.exists(X_path):
        X = np.load(X_path) 
    else:
        X = generate_X(
            num_samples=args.num_samples,
            num_vectors=args.num_vectors,
            dim_vectors=args.dim_vectors,
        )
        np.save(file=X_path, arr=X)

    y = generate_y(X, aggr_method=args.method)
    np.save(file=y_path, arr=y)
