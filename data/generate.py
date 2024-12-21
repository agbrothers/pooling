import os
import argparse
import numpy as np
from tqdm import trange


def softmax(x, axis=None):
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def norm_weighted_avg(x, axis=None):
    """
    Take the weighted avg of each vector in a sample, where 
    the weights are determined by the softmax norm of each vector. 
    
    """
    weights = softmax(np.linalg.norm(x, axis=-1, keepdims=True), axis=axis)
    return np.sum(weights*x, axis=axis)


def subset(x, axis=None):
    """
    Take the weighted avg of each vector in a sample, where 
    the weights are determined by the norm/magn of each vector. 

    Mask any vectors with norms below the 75th percentile in that 
    sample. This is akin to ignoring/attenuating all entities that
    are a threshold distance away from oneself. 
    
    """
    ## MASK ALL BUT VECTORS WITH THE TOP 75th PERCENTILE NORMS
    weights = np.linalg.norm(x, axis=-1, keepdims=True)
    threshold = np.percentile(weights, q=75, axis=1, keepdims=True)
    mask = weights >= threshold
    return np.mean(x, axis=axis, where=mask)


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


def combo(x, axis=None):
    i = x.shape[-1]//4
    return np.hstack((
        np.mean(x[...,   :i*1], axis=axis),
        np.max(x[...,   i:i*2], axis=axis),
        np.min(x[..., 2*i:i*3], axis=axis),
        subset(x[..., 3*i:i*4], axis=axis),
    ))
    

def generate_X(
        num_samples:int, 
        num_vectors:int, 
        dim_vectors:int, 
    ):

    norm = np.sqrt(dim_vectors)
    num_distributions = 3
    dataset_shape = (num_samples, num_vectors, dim_vectors)
    # sample_shape = (dim_vectors//num_distributions+1, num_vectors)
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

        ## NORMALIZE SAMPLE BY DIMENSIONALITY
        

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
        "kfn1": kfn(1),
        "kfn2": kfn(2),
        "kfn3": kfn(3),
        "kfn4": kfn(4),
        "kfn8": kfn(8),
        "kfn16": kfn(16),
        "combo": combo,
        "subset": subset,
        "std": np.std,
    }
    return METHODS[aggr_method](X, axis=1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--method', default="knn16", help='Aggregation method for the dataset.')
    parser.add_argument('-n', '--num_samples', default=1_000_000, help='Number of samples in the dataset.')
    parser.add_argument('-k', '--num_vectors', default=32, help='Number of vectors to aggregate per sample.')
    parser.add_argument('-d', '--dim_vectors', default=16, help='Dimensionality of the vectors per sample.')
    parser.add_argument('-s', '--seed', default=42, help='Seed for dataset generation.')
    args = parser.parse_args()
    np.random.seed(args.seed)

    X_path = os.path.join(os.path.dirname(__file__), f"X{args.dim_vectors}.npy")
    y_path = os.path.join(os.path.dirname(__file__), f"y{args.dim_vectors}_{args.method}.npy")

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
