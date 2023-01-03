import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    path = "/Users/miaha/OneDrive/Dokumenter/DTU/Msc/MLOPS/dtu_mlops_working/data/corruptmnist"

    ## load all training datasets
    train_images = []
    train_labels = []

    for i in range(0,5):
        with np.load(path+'train_' +str(i)+'.npz') as f:
            images_train, labels_train = f['images'], f['labels']
            images_train_scaled = np.array([img/255. for img in images_train])
            train_images.append(torch.from_numpy(images_train_scaled))
            train_labels.append(torch.from_numpy(labels_train))
    
    train = (torch.cat(train_images,dim=0), torch.cat(train_labels,dim=0))

    # Load the validation data 
    with np.load(path+'test.npz') as f:
            images_test, labels_test = f['images'], f['labels']
            images_test_scaled = np.array([img/255. for img in images_test])
            test = (torch.from_numpy(images_test_scaled), torch.from_numpy(labels_test))

    return train, test

mnist()