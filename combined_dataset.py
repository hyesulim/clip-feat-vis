#%%
"""
CelebA dataset ('/home/nas2_userH/hyesulim/Data/celebA') consists of

img_align_celeba  
list_attr_celeba.csv  
list_bbox_celeba.csv  
list_eval_partition.csv  
list_landmarks_align_celeba.csv

In total : 202,599 images
"""

ROOT = '/home/nas2_userH/hyesulim/Data'

#%%
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
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

#%%
attributes = pd.read_csv(f'{ROOT}/celebA/list_attr_celeba.csv')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

celeba_dataset = CelebADataset(image_dir=f'{ROOT}/celebA/img_align_celeba/img_align_celeba', 
                               attributes=attributes, 
                               transform=transform)

imagenet_dataset = ImageNetDataset(image_dir=f'{ROOT}/ImageNet-1K/val_images', transform=transform)


num_samples = 1000  # Number of samples to select
indices = np.random.choice(len(celeba_dataset), num_samples, replace=False)
celeba_sampled_subset = Subset(celeba_dataset, indices)

indices = np.random.choice(len(imagenet_dataset), num_samples, replace=False)
imagenet_sampled_subset = Subset(imagenet_dataset, indices)

combined_dataset = CombinedDataset(celeba_sampled_subset, imagenet_sampled_subset)
combined_dataloader = DataLoader(combined_dataset, batch_size=5, shuffle=True)
#%%

def plot_images(dataloader):
    images, labels = next(iter(dataloader))

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for i in range(5):
        # Convert the tensor to a NumPy array and transpose it
        img = images[i].numpy().transpose(1, 2, 0)
        label = 'CelebA' if labels[i] == 0 else 'ImageNet'

        # Display the image
        ax[i].imshow(img)
        ax[i].set_title(label)
        ax[i].axis('off')  # Turn off axis numbers and labels

    plt.show()
# %%
plot_images(combined_dataloader)

# %%
