import torch
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, preprocess_fn=None):
        # These are now lists of numpy arrays
        self.images = images
        self.masks = masks
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Access the pre-loaded image and mask
        image = self.images[idx]
        mask = self.masks[idx]
        
        # 2. Apply preprocessing (if provided by smp)
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        
        # 3. Convert to Tensor
        # Check if shape is (H, W, C) -> transpose to (C, H, W)
        if image.ndim == 3:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            image_tensor = torch.from_numpy(image).float()
            
        # 4. Handle Mask
        # If mask is 2D (H, W), add channel dimension -> (1, H, W)
        if mask.ndim == 2:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(mask).float()
        
        # Ensure mask is normalized (if it is 0-255)
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
            
        return image_tensor, mask_tensor