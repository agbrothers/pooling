import math
import torch
import pickle


def convert_size(size_bytes:int) -> str:

    if size_bytes == 0: 
        return "0B"

    size_labels = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    size = round(size_bytes / p, 2)
    
    return f"{size}{size_labels[i]}"

def get_size(object_:object) -> str:
    return convert_size(len(pickle.dumps(object_)))    

def get_gpu_memory() -> int:
    return torch.cuda.max_memory_allocated(device=None)
