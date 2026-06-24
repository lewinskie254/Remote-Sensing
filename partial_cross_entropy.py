import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCELoss(nn.Module):
    def __init__(self, focal_gamma=2.0):
        super(PartialCELoss, self).__init__()
        self.focal_gamma = focal_gamma


    def pfce_loss(self, pred, target, mask_labeled):
        """
        Args:
            pred: Model predictions
            target: Ground truth
            mask_labeled: Binary mask where 1 represents labeled pixels
        """
        # Calculate standard focal loss (Just standard cross entropy loss) 
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply the mask 
        masked_loss = ce_loss * mask_labeled
        
        # Calculate the mean based on the number of labeled pixels
        # Adding a small epsilon to avoid division by zero error 
        loss = torch.sum(masked_loss) / (torch.sum(mask_labeled) + 1e-8)
        
        return loss