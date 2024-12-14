import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ChineseMNISTdataset(Dataset):
    def __init__(self, annotations_dataframe, img_dir, transform=None):
        self.df = annotations_dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.img_dir, self.df['file'][idx]) 
        img = Image.open(imgPath)

        # Ensure the image is in grayscale
        img = img.convert('L')  # Convert to grayscale if it's not already
        
        # Apply transformations if they are provided
        if self.transform:
            img = self.transform(img)
        
        # Convert image to tensor format
        imgAsT = torch.from_numpy(np.array(img)).float()

        # If the image is 2D (grayscale), add the channel dimension
        if imgAsT.ndimension() == 2:
            imgAsT = imgAsT.unsqueeze(0)  # Add channel dimension (shape becomes [1, 64, 64])

        # Normalize the image
        imgAsT = imgAsT / 255.0  # Normalize to [0, 1] if necessary (optional based on your transform)

        # Retrieve the label (code)
        label = self.df['code'][idx] - 1  # Subtract 1 to start the labels from 0

        return imgAsT, label