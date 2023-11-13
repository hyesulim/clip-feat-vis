
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_combined_loader(
    root_dir='/home/nas2_userH/hyesulim/Data', 
    batch_size=5, 
    subset_samples=1000,
    transform=None
):
    attributes = pd.read_csv(f'{root_dir}/celebA/list_attr_celeba.csv')

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    celeba_dataset = CelebADataset(image_dir=f'{root_dir}/celebA/img_align_celeba/img_align_celeba', 
                                   attributes=attributes, 
                                   transform=transform)

    imagenet_dataset = ImageNetDataset(image_dir=f'{root_dir}/ImageNet-1K/val_images', transform=transform)
    

    indices = np.random.choice(len(celeba_dataset), subset_samples, replace=False)
    celeba_sampled_subset = Subset(celeba_dataset, indices)

    indices = np.random.choice(len(imagenet_dataset), subset_samples, replace=False)
    imagenet_sampled_subset = Subset(imagenet_dataset, indices)

    combined_dataset = CombinedDataset(celeba_sampled_subset, imagenet_sampled_subset)
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return combined_dataloader

class CelebADataset(Dataset):
    def __init__(self, image_dir, attributes, transform=None):
        self.image_dir = image_dir
        self.attributes = attributes
        self.transform = transform

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.attributes.iloc[idx, 0])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        # Extract attributes or labels if needed
        # labels = self.attributes.iloc[idx, 1:]

        return image  #, labels if using labels

class ImageNetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class CombinedDataset(Dataset):
    def __init__(self, celeba_dataset, imagenet_dataset):
        self.celeba_dataset = celeba_dataset
        self.imagenet_dataset = imagenet_dataset

        # Assuming both datasets are of the same length
        self.dataset_length = len(self.celeba_dataset)

    def __len__(self):
        return self.dataset_length * 2  # Two datasets

    def __getitem__(self, idx):
        if idx < self.dataset_length:
            # CelebA dataset
            return self.celeba_dataset[idx], 0  # Label 0 for CelebA
        else:
            # ImageNet dataset
            return self.imagenet_dataset[idx - self.dataset_length], 1  # Label 1 for ImageNet