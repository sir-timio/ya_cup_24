import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, criterion, weights, device="cuda"):
        super(WeightedLoss, self).__init__()
        self.weights = torch.tensor(weights).float().unsqueeze(0).to(device)
        self.criterion = criterion

    def forward(self, input, target):
        loss = self.criterion(input, target)
        loss = loss * self.weights
        return loss.mean()
