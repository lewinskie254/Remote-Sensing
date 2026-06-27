import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialBCELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, pred, target):

        valid = (target != self.ignore_index).float()
        target = target.clone()
        target[target == self.ignore_index] = 0
        loss = self.bce(pred, target.float())
        loss = (loss * valid).sum() / (valid.sum() + 1e-8)

        return loss