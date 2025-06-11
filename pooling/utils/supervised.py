import os
import time
import math
import yaml
import glob
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed):
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ 
        HUGGING FACE IMPLEMENTATION 
        Source: https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104 
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_acc(criterion, output, batch_y):
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        preds = (torch.sigmoid(output) > 0.5).float()  
        return (preds == batch_y).sum().item() / len(batch_y)
    elif isinstance(criterion, nn.CrossEntropyLoss):
        preds = torch.argmax(output, dim=1) 
        return (preds == batch_y).sum().item() / len(batch_y)
    else:
        return 0.0
    

class ContiguousSplit:

    def __init__(self, n_splits, random_state=42):
        self.n_splits = n_splits

    def split(self, X):
        train_size = len(X)
        fold_size = train_size // self.n_splits
        fold_idx_pairs = []
        for i in range(self.n_splits):
            vL = i * fold_size
            vR = vL + fold_size
            train_idxs_left = list(range(0, vL)) if vL>0 else []
            train_idxs_right = list(range(vR, train_size)) if vR<train_size else []
            train_idxs = train_idxs_left + train_idxs_right
            val_idxs   = list(range(vL, vR))
            fold_idx_pairs.append((train_idxs, val_idxs))
        return iter(fold_idx_pairs)
