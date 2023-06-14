import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler
import pytorch_resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
#from util.util_smote import SMOTE

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS,
    sampler = None
):

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

def create_dataloaders_with_smote_aug(train_dir: str, 
                                test_dir: str, 
                                transform: transforms.Compose, 
                                batch_size: int, 
                                num_workers: int=NUM_WORKERS,
                                sampling_strategy="auto",
                                preprocess: bool = False
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    # Create zero tensors to be augmented
    train_data_to_be_aug = torch.zeros([len(train_data), *list(train_data[0][0].shape)])
    train_label_to_be_aug = torch.zeros(len(train_data))

                                  
    # Get train_data as a tensor
    #for i in range(len(train_data)):
    for i in range(len(train_data)):
        train_data_to_be_aug[i], train_label_to_be_aug[i] = train_data.__getitem__(i)

    # Flatten the image
    train_data_to_be_aug = train_data_to_be_aug.reshape(len(train_data_to_be_aug), -1)

    smt = SMOTE(sampling_strategy=sampling_strategy)  
    train_data_aug, train_label_aug = smt.fit_resample(train_data_to_be_aug, train_label_to_be_aug)
    
    counter = Counter(train_label_aug)
    print(counter)

    # Undersample
    under_sampler = RandomUnderSampler(sampling_strategy=1.0)
    train_data_aug, train_label_aug = under_sampler.fit_resample(train_data_aug, train_label_aug)

    counter = Counter(train_label_aug)
    print(counter)

    # Reshape flattened image to RGB image
    train_data_aug = train_data_aug.reshape(-1, *list(train_data[0][0].shape))
    
    train_data_aug = torch.from_numpy(train_data_aug)
    train_label_aug = torch.from_numpy(train_label_aug)
    train_data = torch.cat((train_data_to_be_aug.reshape(-1, *list(train_data[0][0].shape)), train_data_aug), dim=0)
    train_label = torch.cat((train_label_to_be_aug, train_label_aug), dim=0)

    train_data = TensorDataset(train_data.float(), train_label.long())

    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names