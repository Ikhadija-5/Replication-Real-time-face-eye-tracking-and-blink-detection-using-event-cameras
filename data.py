import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from torchvision.transforms import ToTensor

class ObjectDetectionDataset(Dataset):
    def __init__(self, file_path, images_folder_path, transform=None):
        self.file = pd.read_csv(file_path)
        #self.file['label'] = 'car'  # Add a 'label' column with value 'car'
        self.images_folder_path = images_folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        filen, start_x, start_y, end_x, end_y, label = row["image"], row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["label"]
        image_path = self.images_folder_path + filen
        image = Image.open(image_path).resize((224, 224))
        h, w = image.size
        start_x = start_x / w
        start_y = start_y / h
        end_x = end_x / w
        end_y = end_y / h
        if self.transform is not None:
            image = self.transform(image)
        image = ToTensor()(image)  # convert image to tensor
        target = torch.tensor([start_x, start_y, end_x, end_y], dtype=torch.float32)
        return image, target