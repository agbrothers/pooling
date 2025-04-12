import os
import time
import yaml
import glob
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, TensorDataset
# torch.set_float32_matmul_precision('high')

from pooling.models.attenuator import Attenuator
# from pooling.nn.gem_attention import GemAttention
from pooling.nn.attention import MultiHeadAttention
from pooling.nn.vit import ViT
from pooling.utils.diagnostics import get_gpu_memory, convert_size


MODELS = {
    "Attenuator": Attenuator,
    "Attention": MultiHeadAttention,
    # "GemAttention": GemAttention,
    "ViT": ViT,
}
LOSSES = {
    "MSE": nn.MSELoss(),
    "CrossEntropy": nn.CrossEntropyLoss(),
    "NLL": nn.NLLLoss(),
    "BCELogits": nn.BCEWithLogitsLoss(reduction="sum"),
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
    exp_name = config['EXPERIMENT_NAME'] 

    ## SET SEED = CURRENT EXPERIMENT NUMBER
    seed = len(glob.glob(os.path.join(path, exp_name+"*"))) + 1
    set_seed(seed)
    log_dir = os.path.join(path, f"{config['EXPERIMENT_NAME']}_{seed}_{timestamp}")
    os.makedirs(log_dir)
    config["LEARNING_PARAMETERS"]["SEED"] = seed

    ## SAVE EXPERIMENT CONFIG
    with open(os.path.join(log_dir, "config.yml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False) 

    ## SAVE INITIAL LOG
    history_path = os.path.join(log_dir, f"history.csv") 
    log(history_path, "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Wall Time")
    return log_dir


def log(path, train_loss, val_loss, train_acc, val_acc, wall_time):
    with open(path, "a") as file:
        file.write(f"{train_loss},{val_loss},{train_acc},{val_acc},{wall_time}\n")    
    return 


def save_checkpoint(path, model, value, ckpt_type="loss", tag=""):
    ## OVERWRITE PREVIOUS BEST CHECKPOINT 
    previous_checkpoint = glob.glob(os.path.join(path, f"{tag}ckpt_{ckpt_type}*.pt"))
    if len(previous_checkpoint) > 0:
        os.remove(previous_checkpoint[0])
    ## SAVE NEW BEST CHECKPOINT
    filepath = os.path.join(path, f"{tag}ckpt_{ckpt_type}_{value:.4f}.pt")
    torch.save(model.state_dict(), filepath)        
    return


def load_checkpoint(model, path, ckpt_type="loss"):
    ## LOAD WEIGHTS
    checkpoint = glob.glob(os.path.join(path, f"ckpt_{ckpt_type}*.pt"))[-1]
    print(f"LOADING BEST WEIGHTS: {checkpoint}")
    device = next(model.parameters()).device
    state_dict = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
    

def load_dataset(experiment_path, config):
    ## PARSE CONFIG
    cardinality = config["LEARNING_PARAMETERS"]["INPUT_CARDINALITY"]
    dim_vectors = config["LEARNING_PARAMETERS"]["INPUT_DIM"]
    target = config["LEARNING_PARAMETERS"]["TARGET"]

    ## LOAD DATA
    data_path = os.path.dirname(os.path.dirname(experiment_path)).replace("experiments", "data")
    X = np.load(os.path.join(data_path, f"X-N{cardinality}-d{dim_vectors}.npy"))   # SHAPE: [batch, num_vectors, dim_vectors]
    y = np.load(os.path.join(data_path, f"y-N{cardinality}-d{dim_vectors}-{target}.npy"))  # SHAPE: [batch, aggregate_vector]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create a full dataset
    return TensorDataset(X, y)


def kfold(
        experiment_path:str,
        config:dict,
        load_dataset,
    ):
    ## PARSE CONFIG
    model_config = config["MODEL_CONFIG"]
    # test_ratio = config["LEARNING_PARAMETERS"]["TEST_RATIO"]
    bs = config["LEARNING_PARAMETERS"]["BATCH_SIZE"]
    k = config["LEARNING_PARAMETERS"]["NUM_FOLDS"]
    debug = config["LEARNING_PARAMETERS"]["DEBUG"]
    results = []

    ## BUILD DATASETS
    train_dataset, test_dataset = load_dataset(experiment_path, config)

    ## KFOLD VARIABLES
    # data_shape = dataset.tensors[0].shape
    train_size = len(train_dataset)
    fold_size = train_size // k

    ## INITIALIZE MODEL
    model_config.update({
        # "num_vectors": data_shape[1],
        "dim_ff": 4*model_config["dim_hidden"],
    })
    model_class = MODELS[model_config["model"]]

    ## TRAIN ON EACH FOLD
    for i in range(k):
        
        ## CONFIGURE LOGGING AND SEED
        log_dir = configure_logger(experiment_path, config)
        config["LEARNING_PARAMETERS"]["LOG_DIR"] = log_dir
        seed = config["LEARNING_PARAMETERS"]["SEED"]

        if seed > i+1: 
            shutil.rmtree(log_dir)
            continue

        ## INITIALIZE NEW MODEL
        model = model_class(**model_config, seed=seed)
        
        ## SET DEVICE 
        initial_gpu_mem = get_gpu_memory()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        ## COMPILE MODEL
        if not debug:
            model = torch.compile(model) # requires PyTorch 2.0
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{parameters:,} LEARNABLE PARAMETERS -> {convert_size(get_gpu_memory()-initial_gpu_mem)}")        
        
        ## LOAD SPLIT
        vL = i * fold_size
        vR = vL + fold_size
        train_idxs_left = list(range(0, vL)) if vL>0 else []
        train_idxs_right = list(range(vR, train_size)) if vR<train_size else []
        train_idxs = train_idxs_left + train_idxs_right
        val_idxs   = list(range(vL, vR))
        train_loader = DataLoader(Subset(train_dataset, train_idxs), batch_size=bs, shuffle=True,  generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4
        val_loader   = DataLoader(Subset(train_dataset, val_idxs),   batch_size=bs, shuffle=False, generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4
        test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False, generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4

        print(f"\nTRAINING FOLD {i+1}/{k}")
        test_loss, test_acc = train(
            train_loader, 
            val_loader, 
            test_loader,
            model,
            config,
        )
        results.append(test_acc)
        # results.append(test_loss)
    
    print(f"{k}-FOLD TEST RESULTS: {np.mean(results):.5f} ±{np.std(results):.5f}")
    return


def train(
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader,
        model: nn.Module,
        config: dict,
    ):
    ## PARSE CONFIG
    lr = config["LEARNING_PARAMETERS"]["LEARNING_RATE"]
    seed = config["LEARNING_PARAMETERS"]["SEED"]
    epochs = config["LEARNING_PARAMETERS"]["EPOCHS"]
    log_dir = config["LEARNING_PARAMETERS"]["LOG_DIR"]
    history_path = os.path.join(log_dir, f"history.csv") 
    device = next(model.parameters()).device

    ## INITIALIZE LOSS AND OPTIMIZER
    criterion = LOSSES[config["LEARNING_PARAMETERS"]["LOSS"]]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(train_loader.dataset)
    m = len(val_loader.dataset)

    ## TRAINING LOOP
    best_loss = torch.inf
    best_acc = 0.
    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch_X, batch_y in tqdm(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            ## INFERENCE
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            ## BACKPROP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            train_loss += loss.item()
            
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                preds = (torch.sigmoid(output) > 0.5).float()  
                train_acc += (preds == batch_y).sum().item()                
            elif isinstance(criterion, nn.CrossEntropyLoss):
                preds = torch.argmax(output, dim=1)
                train_acc += (preds == batch_y).sum().item()                

        ## VALIDATION
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_loader.dataset.dataset.train = False
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    preds = (torch.sigmoid(output) > 0.5).float()  
                    val_acc += (preds == batch_y).sum().item()    
                elif isinstance(criterion, nn.CrossEntropyLoss):
                    preds = torch.argmax(output, dim=1)
                    val_acc += (preds == batch_y).sum().item()                

        val_loader.dataset.dataset.train = True
        wall_time = time.time() - start
        epoch_train_loss = train_loss / n
        epoch_train_acc = train_acc / n
        epoch_val_loss = val_loss / m
        epoch_val_acc = val_acc / m
        if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
            print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        else:    
            print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        log(history_path, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc, wall_time)
        
        ## SAVE CHECKPOINT
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(log_dir, model, value=epoch_val_loss, ckpt_type="loss")
        if epoch_val_acc > best_acc:
            best_loss = epoch_val_loss
            save_checkpoint(log_dir, model, value=epoch_val_acc, ckpt_type="acc")

    ## LOAD CHECKPOINT WITH BEST VALIDATION LOSS
    model = load_checkpoint(model, log_dir, ckpt_type="acc")

    ## TESTING
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    # test_loader.dataset.dataset.train = False
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
        
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                preds = (torch.sigmoid(output) > 0.5).float()  
                test_acc += (preds == batch_y).sum().item()    
            elif isinstance(criterion, nn.CrossEntropyLoss):
                preds = torch.argmax(output, dim=1)
                test_acc += (preds == batch_y).sum().item()                


    ## SAVE THE FINAL MODEL
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    log(history_path, "TEST LOSS", test_loss, "TEST ACC", test_acc, 0.0)
    if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
        print(f"\nEXPERIMENT COMPLETE | Best Val Loss: {best_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
        save_checkpoint(log_dir, model, value=test_acc, ckpt_type="acc", tag="final_")
    else:    
        print(f"\nEXPERIMENT COMPLETE | Best Val Loss: {best_loss:.4f} | Test Loss: {test_loss:.4f}\n")
        save_checkpoint(log_dir, model, value=test_loss, ckpt_type="loss", tag="final_")

    return test_loss, test_acc



if __name__ == "__main__":

    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/gem-baseline/mix-gem", help='Relative path to the experiment config.')
    args = parser.parse_args()

    ## BUILD PATHS FROM ARGPARSE INPUT
    configs = []
    base = __file__.split("train")[0]
    abs_path = os.path.join(base, args.experiment_path)
    
    ## CHECK IF THIS PATH LEADS TO A SINGLE EXPERIMENT OR A SET
    is_single_exp = os.path.exists(os.path.join(abs_path, "config.yml"))
    paths = [abs_path] if is_single_exp else glob.glob(os.path.join(abs_path, "*"))

    ## LOAD EXPERIMENT CONFIG(S)
    for path in paths:
        with open(os.path.join(base, path, "config.yml"), "r") as file:
            configs.append(yaml.safe_load(file))

    ## K-FOLD CROSS VALIDATION
    for path,config in list(zip(paths, configs)):
        target = os.path.basename(path).upper()
        path = os.path.join(base, path)
        for i in range(config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]):
            print(f"\nSTARTING EXPERIMENT FOR GEM APPROXIMATION OF {config['EXPERIMENT_NAME']}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
            kfold(path, config, load_dataset)
