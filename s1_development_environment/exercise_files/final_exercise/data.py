import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    train = np.load("data/corruptmnist/train_0.npz")

    # Convert data from numpy to torch 
    images = torch.from_numpy(train["images"])
    # Flatten the images 
    images = images.view(5000, 784)

    labels = torch.from_numpy(train["labels"])

    # Load the validation data 
    test = torch.randn(10000, 784) 
    return train, test

mnist()