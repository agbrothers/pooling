### ***Robust Noise Attenuation via***
# ADAPTIVE POOLING OF TRANSFORMER OUTPUTS

This repository contains the code accompanying the research paper [Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](https://arxiv.org/abs/2506.09215). It contains all of the original code used to run the experiments described in the paper. See below for installation instructions and an overall description of the repo. 

<br>

***NOTE**: All code was developed and tested on Linux (Ubuntu 22.04), using the `uv` python package manager.*

## INSTALL
> 1. Download uv
> ```
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
> ```
> source ~/.bashrc
> ```
>
> 2. Clone this repo and set up the venv
> ```
> git clone https://github.com/agbrothers/pooling.git
> ```
> ```
> cd pooling; uv venv --python 3.11.11 --seed; source .venv/bin/activate
> ```
> ```
> UV_HTTP_TIMEOUT=1000 uv sync --active
> ```
> 3. [Optional] Append the following lines to your .bashrc for convenience
> ```
> export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
> export UV_HTTP_TIMEOUT=1000 
> alias env="source .venv/bin/activate"
> alias pooling=cd path/to/pooling>; env"
> ```
> Then source it for the changes to take effect:
> ```
> source ~/.bashrc
> ```
> Typing `pooling` in the terminal will now automatically navigate to the repo and activate the uv python environment. If you already have conda installed, you may want to add `conda deactivate` to the alias. 
> 
> Note: All experiments were run with Python 3.11.11, and the versions of all packages installed during experimentation can be found in requirements.txt if any versioning issues are encountered. The version of ray (`2.37.0`) is particularly important for reproducing the RL experiments, as there may be breaking changes in older or more recent versions. Exact versions for all packages used for experiments can be found in the `requirements.txt`. 

<br>


## EXPERIMENTS 

> ### RUNNING CIFAR EXPERIMENTS
> All experiment configs for CIFAR can be found in `pooling/experiments/cifar10` and `pooling/experiments/cifar100`. The datasets will be automatically downloaded the first time the following training script is run. As an example, to reproduce the experiment on CIFAR 100 using AdaPool with a focal query, run the following:
>
> ```
> python train/train_supervised.py -p experiments/cifar100/ada-focal
> ```
>
> If you have multiple GPUs, to specify index of the GPU used to run the experiment, i.e. GPU 2, prefix the the above line with the environment varaible `CUDA_VISIBLE_DEVICES=2`. 

<br/>

> ### RUNNING REINFORCEMENT LEARNING EXPERIMENTS
> Similar to the supervised experiments, all RL experiment configs can be found in `experiments/simple-centroid`, `experiments/simple-tag`, `experiments/boxworld-entities`, and `experiments/boxworld-pixels`. To run an RL experiment, simply run find the experiment directory you would like to reproduce, such as Simple Centroid 1v3v4, and run: 
> 
> ```
> python train/train_rl.py -p=experiments/simple-tag/1v3v4/ada/
> ```
> Note that by default, the config utilizes and 16 parallel workers to collect experience. You may need to reduce this `NUM_WORKERS` parameter depending on compute constraints. 
> 
> Videos, results, and tensorboard logs are stored adjacent to the config at the filepath specified in the line above. Results can be visualized live by running the tensorboard server. To do so, navigate to the pooling directory, activate the uv environment, then kickoff tensorboard via `tensorboard --logdir='./experiments' --host 0.0.0.0 --port 6008`. To see the results, navigate to `http://hostname:6008/` in your browser, where hostname is the name or IP of the machine running the tensorboard server. Once experiments are finished, final plots can be generated using the scripts in the `analysis` directory. 

<br/>

> ### RUNNING KNN-CENTROID SUPERVISED EXPERIMENTS
> We include a script for generating the synthetic dataset in `pooling/datasets/knn_centroid/dataset.py`. 
> 
> To generate an input-output pair for a specific signal-to-noise ratio, run the script with the following arguments:
> ```
> python pooling/datasets/knn_centroid/dataset.py --method="knn32" --num_samples=1000000 --num_vectors=128 --dim_vectors=16 --seed=42
> ```
> 
> The input dataset X and any targets will be saved in the `pooling/datasets/knn_centroid` directory. To create all data needed to reproduce the synthetic dataset experiment as presented in the paper, use the above command for each of the following methods: `knn1`, `knn2`, `knn4`, `knn8`, `knn16`, `knn32`, `knn64`, and `knn128`.
> 
> The configs for running each experiment on this dataset can be found in `experiments/noise-robustness`. To run an individual experiment using AdaPool on the knn32 data generated above, run the following:
> 
> ```
> python train/train_supervised.py -p=experiments/noise-robustness/knn32/ada
> ```
> 
> As indicated by the config in that experiment directory, this will run 5-fold cross-validation with 100 epochs per fold for that single method on the 32-neighbor task. Results for each fold will be saved in a separate subdirectory adjacent to the experiment config. Plotting tools can be found in `pooling/analysis` to visualize results. 

<br/>

## CODE

> ### NEURAL NETWORK IMPLEMENTATIONS
> All code relating to neural network components is located in `pooling/nn`, and the single architecture used across all experiments can be found in `pooling/models/attenuator.py`. All pooling methods are located in `pooling/nn/pool.py`. All implementations are in PyTorch. 
> 
>Wrappers for compatibility and registration with RLlib can be found in `pooling/wrappers`

<br/>

> ### TRAINING ENVIRONMENTS
> We have included the custom simple centroid scenario for the Multi-Particle Environment in `pooling/envs/mpe_centroid.py`. 
> 
> We have a local implementation of BoxWorld in `pooling/envs/boxworld.py`, forked from Nathan Grinsztajn's open-source implementation (https://github.com/nathangrinsztajn/Box-World). 


## Bibtex
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
