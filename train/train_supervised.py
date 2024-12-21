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

from pooling.models.attenuator import Attenuator
from pooling.models.autoencoder import Autoencoder
from pooling.models.aggregation_mlp import AggregationMLP
from pooling.utils.diagnostics import get_gpu_memory, convert_size


MODELS = {
    "Attenuator": Attenuator,
    "Autoencoder": Autoencoder,
    "AggregationMLP": AggregationMLP
}


def set_seed(seed):
    print(f" • seed = {seed}\n")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable to ensure deterministic behavior
    return


def configure_logger(path, config):
    ## CREATE LOGGING DIRECTORY
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_method = config["MODEL_CONFIG"].get("pooling_method", "mlp")
    experiment_name =  f"{model_method}_{timestamp}"
    log_dir = os.path.join(path, experiment_name)
    os.makedirs(log_dir)

    ## SET SEED = CURRENT EXPERIMENT NUMBER
    dirname = os.path.dirname(log_dir)
    method = os.path.basename(log_dir).split("_")[0]
    seed = len(glob.glob(os.path.join(dirname, method+"*")))
    set_seed(seed)
    config["LEARNING_PARAMETERS"]["SEED"] = seed

    ## SAVE EXPERIMENT CONFIG
    with open(os.path.join(log_dir, "config.yml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False) 

    ## SAVE INITIAL LOG
    history_path = os.path.join(log_dir, f"history.csv") 
    log(history_path, "Train Loss", "Val Loss", "Wall Time")
    return log_dir, history_path


def log(path, train_loss, val_loss, wall_time):
    with open(path, "a") as file:
        file.write(f"{train_loss},{val_loss},{wall_time}\n")    
    return 


def save_checkpoint(path, model, loss, tag=""):
    ## OVERWRITE PREVIOUS BEST CHECKPOINT 
    previous_checkpoint = glob.glob(os.path.join(path, f"ckpt_{tag}*.pt"))
    if len(previous_checkpoint) > 0:
        os.remove(previous_checkpoint[0])
    ## SAVE NEW BEST CHECKPOINT
    filepath = os.path.join(path, f"{tag}ckpt_{loss:.4f}.pt")
    torch.save(model.state_dict(), filepath)        
    return


def load_checkpoint(model, path):
    ## LOAD WEIGHTS
    checkpoint = glob.glob(os.path.join(path, "ckpt_*.pt"))[-1]
    print(f"LOADING BEST WEIGHTS: {checkpoint}")
    device = next(model.parameters()).device
    state_dict = torch.load(checkpoint, weights_only=True, map_location=device)

    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model
    


def load_dataset(experiment_path, dim_vectors, seed):
    # Load data
    method = os.path.basename(experiment_path)
    data_path = os.path.dirname(experiment_path).replace("experiments", "data")
    X = np.load(os.path.join(data_path, f"X{dim_vectors}.npy"))   # SHAPE: [batch, num_vectors, dim_vectors]
    y = np.load(os.path.join(data_path, f"y{dim_vectors}_{method}.npy"))  # SHAPE: [batch, aggregate_vector]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create a full dataset
    full_dataset = TensorDataset(X, y)

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


def train(
        experiment_path:str,
        config:dict,
        load_dataset,
    ):
    ## PARSE CONFIG
    model_config = config["MODEL_CONFIG"]
    dim_vectors = config["LEARNING_PARAMETERS"]["DIM_INPUT"]
    epochs = config["LEARNING_PARAMETERS"]["EPOCHS"]
    debug = config["LEARNING_PARAMETERS"]["DEBUG"]
    bs = config["LEARNING_PARAMETERS"]["BATCH_SIZE"]
    lr = config["LEARNING_PARAMETERS"]["LEARNING_RATE"]

    ## CONFIGURE LOGGING AND SEED
    log_dir, history_path = configure_logger(experiment_path, config)
    seed = config["LEARNING_PARAMETERS"]["SEED"]

    ## BUILD DATASETS
    train_dataset, val_dataset, test_dataset = load_dataset(experiment_path, dim_vectors, seed)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    ## INITIALIZE MODEL
    X_shape = train_dataset.dataset.tensors[0].shape
    y_shape = train_dataset.dataset.tensors[1].shape
    model_config.update({
        "num_vectors": X_shape[1],
        "dim_ff": 4*model_config["dim_hidden"],
        # "dim_hidden": X_shape[2],
    })
    model_class = MODELS[model_config["model"]]
    model = model_class(**model_config)
    
    ## SET DEVICE 
    initial_gpu_mem = get_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## COMPILE MODEL
    if not debug:
        model = torch.compile(model) # requires PyTorch 2.0
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{parameters:,} LEARNABLE PARAMETERS -> {convert_size(get_gpu_memory()-initial_gpu_mem)}")        
    
    ## INITIALIZE LOSS AND OPTIMIZER
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## TRAINING LOOP
    best_loss = torch.inf
    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            ## INFERENCE
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            ## BACKPROP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            train_loss += loss.item()
        
        ## VALIDATION
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                # dist = torch.mean(torch.linalg.norm(predictions-batch_y, axis=1))
                val_loss += loss.item()
                # val_dist += dist.item()

        wall_time = time.time() - start
        # epoch_val_dist = val_dist / len(val_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_loss = train_loss / len(train_loader)
        # print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Dist {epoch_val_dist:.4f}")
        print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        log(history_path, epoch_train_loss, epoch_val_loss, wall_time)
        
        ## SAVE CHECKPOINT
        # if epoch > int(0.05*epochs) and epoch_val_loss < best_loss:
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(log_dir, model, epoch_val_loss)

    
    ## LOAD CHECKPOINT WITH BEST VALIDATION LOSS
    model = load_checkpoint(model, log_dir)

    ## TESTING
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

    ## SAVE THE FINAL MODEL
    test_loss = test_loss / len(test_loader)
    log(history_path, "TEST LOSS", test_loss, 0.0)
    print(f"\nEXPERIMENT COMPLETE | Best Val Loss: {best_loss:.4f} | Test Loss: {test_loss:.4f}\n")
    save_checkpoint(log_dir, model, test_loss, tag="final_")
    return



if __name__ == "__main__":

    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/knn1", help='Relative path to the experiment config.')
    args = parser.parse_args()

    # paths = [
    #     "./experiments/max",
    #     "./experiments/min",
    #     "./experiments/mean",
    #     "./experiments/subset",
    #     "./experiments/combo",
    # ]
    # base = __file__.split("train")[0]
    # configs = []
    # for path in paths:
    #     with open(os.path.join(base, path, "config.yml"), "r") as file:
    #         configs.append(yaml.safe_load(file))

    # for path,config in list(zip(paths, configs)):
    #     ## TRAIN
    #     method = config["MODEL_CONFIG"].get("pooling_method", "mlp").upper()
    #     target = os.path.basename(path).upper()
    #     for i in range(config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]):
    #         print(f"\nSTARTING EXPERIMENT FOR {method} POOLING APPROXIMATION OF {target}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
    #         train(path, config, load_dataset)


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
        train(path, config, load_dataset)
