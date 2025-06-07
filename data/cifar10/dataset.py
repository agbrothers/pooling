import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

"""
HYPERPARAMETERS ADOPTED FROM - https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py

"""


class KFoldDataset:

    def __init__(self, root, train=True, train_transform=None, test_transform=None, target_transform=None):
        self.dataset = CIFAR10(train, train_transform, target_transform, download=False)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train = True
        return
    
    def __getitem__(self, index):
        return self.dataset



class CIFAR10_KFOLD(CIFAR10):
    def __init__(self, root, train=True, transform=None, test_transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.test_transform = test_transform
        self.train = True
        return

    def __getitem__(self, index):
        """
        Modify generic CIFAR dataset to enable splitting compatible with 
        our implementation of K-Fold cross validation.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.train and self.transform is not None:
            img = self.transform(img)
        elif not self.train and self.test_transform is not None:
            img = self.test_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    

def load_cifar_10(experiment_path, config):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010) 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cifar10")
    trainset = CIFAR10_KFOLD(root=path, train=True, transform=transform_train, test_transform=transform_test)
    testset = CIFAR10_KFOLD(root=path, train=False, transform=transform_test)
    return trainset, testset   
