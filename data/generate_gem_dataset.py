import os
import argparse
import numpy as np
from numpy import log, exp, sum
from tqdm import trange


def generalized_mean(x, p=1, axis=-2):
    return np.mean(x**p, axis=axis) ** (1/p)

def mix(x, axis=-2):
    ## COMPUTE P
    d = x.shape[-1] // 2
    p = np.arange(-d, d, 1).astype(float) + 1
    p[d-1] = 1e-6
    p = p[None, None, :]

    ## LSE TRICK
    z = p * log(x)
    Z_max = np.max(z, axis=axis, keepdims=True)
    return f_inv( np.mean( f(z, Z_max), axis=axis, keepdims=True), p, Z_max )

def harmonic_mean(x, axis=-2):
    ## p = -1
    n = x.shape[axis]
    return n / sum(1/x, axis=axis, keepdims=True)

def geometric_mean(x, axis=-2):
    ## p = 0
    n = x.shape[axis]
    return np.prod(x, axis=axis, keepdims=True) ** (1/n)

def arithmetic_mean(x, axis=-2):
    ## p = 1
    return np.mean(x, axis=axis, keepdims=True)

def root_mean_square(x, axis=-2):
    ## p = 2
    return np.mean(x**2, axis=axis, keepdims=True) ** (1/2)

def cubic_mean(x, axis=-2):
    ## p = 3
    return np.mean(x**3, axis=axis, keepdims=True) ** (1/3)

def maximum(x, axis=-2):
    ## p = inf
    return np.max(x, axis=axis, keepdims=True) 

def minimum(x, axis=-2):
    ## p = -inf
    return np.min(x, axis=axis, keepdims=True) 

def std(x, axis=-2):
    ## p = -inf
    return np.std(x, axis=axis, keepdims=True) 
    

def f(z, Z_max):
    return exp( z - Z_max )

def f_inv(y, p, Z_max):
    return exp( 1/p * (Z_max + log(y)) )



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
        )).T 
        sample -= sample.min(axis=-1, keepdims=True)
        sample += np.random.uniform(0, 1, sample.shape[0])[:, None]
        sample /= norm

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
        "min": np.min,
        "harmonic": harmonic_mean,
        "geometric": geometric_mean,
        "arithmetic": np.mean,
        "rms": root_mean_square,
        "cubic": cubic_mean,
        "max": np.max,
        "gem": generalized_mean,
        "mix": mix,
        "std": std,
    }
    return METHODS[aggr_method](X, axis=1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--method', default="rms", type=str, help='Aggregation method for the dataset.')
    parser.add_argument('-n', '--num_samples', default=1_000_000, type=int, help='Number of samples in the dataset.')
    parser.add_argument('-k', '--num_vectors', default=32, type=int, help='Number of vectors to aggregate per sample.')
    parser.add_argument('-d', '--dim_vectors', default=16, type=int, help='Dimensionality of the vectors per sample.')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Seed for dataset generation.')
    args = parser.parse_args()
    np.random.seed(args.seed)

    X_path = os.path.join(os.path.dirname(__file__), f"X-N{args.num_vectors}-d{args.dim_vectors}.npy")
    y_path = os.path.join(os.path.dirname(__file__), f"y-N{args.num_vectors}-d{args.dim_vectors}-{args.method}.npy")

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
