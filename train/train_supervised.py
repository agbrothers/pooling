import os
import time
import yaml
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from pooling.nn.vit import ViT
from pooling.nn.attention import MultiHeadAttention
from pooling.models.attenuator import Attenuator
from pooling.utils.diagnostics import get_gpu_memory, convert_size
from pooling.utils.supervised import (
    ContiguousSplit, 
    load_checkpoint, 
    save_checkpoint, 
    get_cosine_schedule_with_warmup,
    get_acc, 
    configure_logger,
    log,
)
from data.knn_centroid.dataset import load_knn_centroid
from data.cifar10.dataset import load_cifar_10
from data.cifar100.dataset import load_cifar_100
from data.imagenet.dataset import load_imagenet


MODELS = {
    "Attenuator": Attenuator,
    "Attention": MultiHeadAttention,
    "ViT": ViT,
}
LOSSES = {
    "MSE": nn.MSELoss(),
    "CrossEntropy": nn.CrossEntropyLoss(),
    "NLL": nn.NLLLoss(),
    "BCELogits": nn.BCEWithLogitsLoss(reduction="sum"),
}
LOADERS = {
    "KNN_CENTROID": load_knn_centroid,
    "CIFAR10": load_cifar_10,
    "CIFAR100": load_cifar_100,
    "IMAGENET": load_imagenet,
}
    

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
    grad_clip = config["LEARNING_PARAMETERS"].get("GRAD_CLIP")
    weight_decay = config["LEARNING_PARAMETERS"].get("WEIGHT_DECAY", 1e-6)
    scheduler_warmup_steps = config["LEARNING_PARAMETERS"].get("SCHEDULER_WARMUP_STEPS")
    use_scheduler = scheduler_warmup_steps is not None
    history_path = os.path.join(log_dir, f"history.csv") 
    device = next(model.parameters()).device

    ## INITIALIZE LOSS AND OPTIMIZER
    criterion = LOSSES[config["LEARNING_PARAMETERS"]["LOSS"]]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    if use_scheduler:
        scheduler = get_cosine_schedule_with_warmup(optimizer, scheduler_warmup_steps, epochs)
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
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()            
            train_loss += loss.item()
            train_acc += get_acc(criterion, output, batch_y)

        if use_scheduler:
            scheduler.step()
            
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
                val_acc += get_acc(criterion, output, batch_y)             

        val_loader.dataset.dataset.train = True
        wall_time = time.time() - start
        epoch_train_loss = train_loss / n
        epoch_train_acc = train_acc / n
        epoch_val_loss = val_loss / m
        epoch_val_acc = val_acc / m
        if isinstance(criterion, nn.MSELoss): 
            print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        else:
            print(f"Exp {seed}. Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        log(history_path, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc, wall_time)
        
        ## SAVE CHECKPOINT
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(log_dir, model, value=epoch_val_loss, ckpt_type="loss")
        if epoch_val_acc > best_acc:
            best_loss = epoch_val_loss
            save_checkpoint(log_dir, model, value=epoch_val_acc, ckpt_type="acc")

    ## TESTING
    if test_loader is None:
        return (None, None)
    
    ## LOAD CHECKPOINT WITH BEST VALIDATION LOSS
    if isinstance(criterion, nn.MSELoss): 
        model = load_checkpoint(model, log_dir, ckpt_type="loss")
    else:
        model = load_checkpoint(model, log_dir, ckpt_type="acc")
    
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
            test_acc += get_acc(criterion, output, batch_y)           

    ## SAVE THE FINAL MODEL
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    log(history_path, "TEST LOSS", test_loss, "TEST ACC", test_acc, 0.0)
    if isinstance(criterion, nn.MSELoss): 
        print(f"\nEXPERIMENT COMPLETE | Best Val Loss: {best_loss:.4f} | Test Loss: {test_loss:.4f}\n")
        save_checkpoint(log_dir, model, value=test_loss, ckpt_type="loss", tag="final_")
    else:
        print(f"\nEXPERIMENT COMPLETE | Best Val Loss: {best_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
        save_checkpoint(log_dir, model, value=test_acc, ckpt_type="acc", tag="final_")
    return test_loss, test_acc
    

def kfold(
        experiment_path:str,
        config:dict,
        load_dataset,
    ):
    ## PARSE CONFIG
    model_config = config["MODEL_CONFIG"]
    bs = config["LEARNING_PARAMETERS"]["BATCH_SIZE"]
    k = config["LEARNING_PARAMETERS"]["NUM_FOLDS"]
    debug = config["LEARNING_PARAMETERS"]["DEBUG"]
    results_loss = []
    results_acc = []

    ## BUILD DATASETS
    train_dataset, test_dataset = load_dataset(experiment_path, config)

    ## GET K-FOLD SPLITS
    split = ContiguousSplit(n_splits=k) 
    fold_idx_pairs = split.split(np.zeros(len(train_dataset)))

    ## INITIALIZE MODEL
    model_config.update({"dim_ff": 4*model_config["dim_hidden"]})
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
        print(f" • seed = {seed}\n")

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
        train_idxs, val_idxs = next(fold_idx_pairs)
        train_loader = DataLoader(Subset(train_dataset, train_idxs), batch_size=bs, shuffle=True,  generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4
        val_loader   = DataLoader(Subset(train_dataset, val_idxs),   batch_size=bs, shuffle=False, generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4
        test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False, generator=torch.Generator().manual_seed(seed)) #num_workers=os.cpu_count()//4
        # t = list(np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices])
        # v = list(np.array(val_loader.dataset.dataset.targets)[val_loader.dataset.indices])
        # t_dist = {int(c):t.count(c) for c in set(t)}
        # v_dist = {int(c):v.count(c) for c in set(v)}

        print(f"\nTRAINING FOLD {i+1}/{k}")
        test_loss, test_acc = train(
            train_loader, 
            val_loader, 
            test_loader,
            model,
            config,
        )
        results_loss.append(test_loss)
        results_acc.append(test_acc)
    
    if len(results_loss) > 0:
        print(f"{k}-FOLD RESULTS | TEST LOSS: {np.mean(results_loss):.5f} ±{np.std(results_loss):.5f} | {np.mean(results_acc)*100:.5f}% ±{np.std(results_acc)*100:.5f}")
    else:
        print(f"/!\\ {k}-Fold results already completed for {config['EXPERIMENT_NAME']}")
    return

 

def run(train_func) -> None:
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-p', '--experiment_path', default="./experiments/imagenet/ada-focal", help='Relative path to the experiment config.')
    parser.add_argument('-p', '--experiment_path', default="./experiments/cifar10/ada-corner", help='Relative path to the experiment config.')
    # parser.add_argument('-p', '--experiment_path', default="./experiments/noise-robustness/knn2", help='Relative path to the experiment config.')
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
        path = os.path.join(base, path)
        loader = LOADERS[config["DATASET_NAME"]]
        for i in range(config["LEARNING_PARAMETERS"]["NUM_EXPERIMENTS"]):
            print(f"\nSTARTING EXPERIMENT {config['EXPERIMENT_NAME']}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
            train_func(path, config, loader)
    return


if __name__ == "__main__":

    run(train_func=kfold)
