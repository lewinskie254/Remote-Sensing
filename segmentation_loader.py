import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, preprocess_fn=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Apply preprocessing
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        
        # Convert to Tensor (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # Add channel dim
        
        # Normalize mask if necessary (e.g., 0-1 instead of 0-255)
        mask_tensor = mask_tensor / 255.0
        
        return image_tensor, mask_tensor