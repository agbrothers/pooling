import os
import cv2
import glob
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# torch.set_float32_matmul_precision('high')

from train_supervised import kfold


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, train_transform=None, test_transform=None):
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train = True
        self.file_list = np.array(sorted(file_list))
        np.random.shuffle(self.file_list)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        img_transformed = self.train_transform(img) if self.train else self.test_transform(img)

        label = os.path.basename(img_path).split(".")[0]
        label = 1 if label == "dog" else 0
        return img_transformed['image'], torch.tensor(label, dtype=torch.long)


class DogCatDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transforms to apply to images.
        """
        self.transform = transform
        self.file_list = []

        # Collect image paths and labels
        for file in file_list:
            if os.path.basename(file).startswith("cat."):
                self.file_list.append((file, 0))  # 0 for cat
            elif os.path.basename(file).startswith("dog."):
                self.file_list.append((file, 1))  # 1 for dog

        self.file_list.sort()  # Optional: Sort for consistent ordering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = np.array(Image.open(img_path).convert("RGB"))  # Ensure RGB format

        if self.transform:
            image = self.transform(image=image)

        return image['image'], torch.tensor(label, dtype=torch.long)


def load_dataset(experiment_path, config):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/cats_and_dogs")
    train_list = glob.glob(os.path.join(path, "train", '*.jpg'))

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ]
    # )
    train_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Move ToTensorV2 to the end of the pipeline
    ])
    test_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Move ToTensorV2 to the end of the pipeline
    ])

    # Create a full dataset
    # return DogCatDataset(train_list+test_list, transform=transform)
    return CatsDogsDataset(train_list, train_transform=train_transform, test_transform=test_transform)


if __name__ == "__main__":

    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--experiment_path', default="./experiments/gem-baseline/cnd-attn", help='Relative path to the experiment config.')
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
            print(f"\nSTARTING EXPERIMENT FOR GEM APPROXIMATION OF {target}:  {i+1}/{config['LEARNING_PARAMETERS']['NUM_EXPERIMENTS']}")
            kfold(path, config, load_dataset)
