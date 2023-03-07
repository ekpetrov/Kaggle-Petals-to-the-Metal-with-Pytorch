from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class Dataset_ptm(Dataset):
    def __init__(self, csv_path, data_path=None, transform=None, augment=False):
        if data_path is None:
            data_path = csv_path.split('/')
            data_path = '/'.join(data_path[:-1])
        self.data_path = data_path
        if transform is None:
            transform = self.get_transform()
        self.transform = transform
        self.augment = augment
        self.df = pd.read_csv(csv_path)

    def get_transform(self):
        # I wanted to calculate mean and std for entire dataset,
        # but this will do for now
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        return transform

    def get_img(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.df.shape[0])
        img = Image.open(
            f'{self.data_path}/{self.df.iloc[idx]["filename"]}'
        ).convert('RGB')
        return(img)
    
    def augment_image(self, img):
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])
        return(aug_transform(img))
    
    def __getitem__(self, idx):
        df_idx = self.df.iloc[idx]
        img = self.get_img(idx)
        c = df_idx['class']
        img = self.transform(img)
        if self.augment:
            img = self.augment_image(img)
        return img, c

    def __len__(self):
        return self.df.shape[0]