import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision import transforms

"""
HYPERPARAMETERS ADOPTED FROM - https://github.com/google-research/vision_transformer

"""

def load_imagenet(experiment_path, config):
    ## [TODO] ADD OPTION TO TRAIN WITH BFLOAT 16?
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imagenet")
    # path = "/awlpool/awlcloud/datasets/imagenet"
    trainset = ImageNet(root=path, split="train", transform=transform_train)
    valset = ImageNet(root=path, split="val", transform=transform_test)
    return trainset, valset
