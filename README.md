# Robust Noise Attenuation via *Adaptive Pooling* of *Transformer Outputs*

This repository contains the code accompanying the research paper [Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](https://arxiv.org/abs/2506.09215). It contains all of the original code used to run the experiments described in the paper. See below for installation instructions and an overall description of the repo. 


### INSTALLATION
Navigate to the root directory and run `pip install .`, or `pip install -e .` for an editable install. 

All experiments were run with Python 3.11.11, and the versions of all packages installed during experimentation can be found in requirements.txt if any versioning issues are encountered. The version of ray (`2.37.0`) is particularly important for reproducing the RL experiments, as there may be breaking changes in older or more recent versions. Exact versions for all packages used for experiments can be found in the `requirements.txt`. 


### NEURAL NETWORK CODE
All code relating to neural network components is located in `pooling/nn`, and the single architecture used across all experiments can be found in `pooling/models/attenuator.py`. All pooling methods are located in `pooling/nn/pool.py`. All implementations are in PyTorch. 

Wrappers for compatibility and registration with RLlib can be found in `pooling/wrappers`


### RUNNING KNN-CENTROID SUPERVISED EXPERIMENTS
We include a script for generating the synthetic dataset in `pooling/datasets/knn_centroid/dataset.py`. 

To generate an input-output pair for a specific signal-to-noise ratio, run the script with the following arguments:
`python pooling/datasets/knn_centroid/dataset.py --method="knn32" --num_samles=1000000 --num_vectors=128 --dim_vectors=16 --seed=42`

The input dataset X and any targets will be saved in the `pooling/datasets/knn_centroid` directory. To create all data needed to reproduce the synthetic dataset experiment as presented in the paper, use the above command for each of the following methods: `knn1`, `knn2`, `knn4`, `knn8`, `knn16`, `knn32`, `knn64`, and `knn128`.

The configs for running each experiment on this dataset can be found in `experiments/noise-robustness`. To run an individual experiment using AdaPool on the knn32 data generated above, run the following:

`python train/train_supervised.py -p=experiments/noise-robustness/knn32/ada`

As indicated by the config in that experiment directory, this will run 5-fold cross-validation with 100 epochs per fold for that single method on the 32-neighbor task. Results for each fold will be saved in a separate subdirectory adjacent to the experiment config. Plotting tools can be found in `pooling/analysis` to visualize results. 


### RUNNING REINFORCEMENT LEARNING EXPERIMENTS
Similar to the supervised experiments, all experiment configs can be found in `experiments/simple-centroid`, `experiments/simple-tag`, `experiments/boxworld-entities`, and `experiments/boxworld-pixels`. To run an RL experiment, simply run find the experiment directory you would like to reproduce, such as Simple Centroid 1v3v28, and run: 

`python train/train_rl.py -p=experiments/simple-tag/simple-tag/1v3v28/ada/`

Again, the experiment results will be stored adjacent to the config used to run it, and results can be visualized using the scripts in the `analysis` directory. Note that by default, the config utilizes and 16 parallel workers to collect experience. You may need to reduce this `NUM_WORKERS` parameter depending on compute constraints. 


### RUNNING CIFAR EXPERIMENTS
1. Download `cifar-10-python.tar.gz` from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, move it to `pooling/datasets/cifar10`, and untar.   
2. Download `cifar-100-python.tar.gz` from `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`, move it to `pooling/datasets/cifar100`, and untar.   

All experiment configs for CIFAR can be found in `pooling/datasets/cifar10` and `pooling/datasets/cifar100`. As an example, to reproduce the experiment on CIFAR 100 using AdaPool with a focal query, run the following:

`python train_cifar100.py -p experiments/cifar100/ada-focal`


### TRAINING ENVIRONMENTS
We have included the custom simple centroid scenario for the Multi-Particle Environment in `pooling/envs/mpe_centroid.py`. 

We have a local implementation of BoxWorld in `pooling/envs/boxworld.py`, forked from Nathan Grinsztajn's open-source implementation (https://github.com/nathangrinsztajn/Box-World). 


### Bibtex
```
@inproceedings{
    brothers2025robust,
    title={Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs},
    author={Greyson Brothers},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=8JGwoZceQs}
}
```
