# Define regression network
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self, zDim):
        super(RegressionModel, self).__init__()
        self.zDim = zDim
        self.FC1 = nn.Linear(in_features = self.zDim, out_features = 2*self.zDim, dtype=float)
        self.FC2 = nn.Linear(in_features = 2*self.zDim, out_features = 4*self.zDim, dtype=float)
        self.FC3 = nn.Linear(in_features = 4*self.zDim, out_features = 2*self.zDim, dtype=float)
        self.FC4 = nn.Linear(in_features = 2*self.zDim, out_features = self.zDim, dtype=float)
        self.FC5 = nn.Linear(in_features = self.zDim, out_features = self.zDim//2, dtype=float)
        self.FC6 = nn.Linear(in_features = self.zDim//2, out_features = 1, dtype=float)
        for w in [self.FC1, self.FC2, self.FC3, self.FC4, self.FC5, self.FC6]:
            torch.nn.init.xavier_normal_(w.weight)

    def forward(self, x):
        x = torch.sigmoid(self.FC1(x))
        x = F.sigmoid(self.FC2(x))
        x = F.sigmoid(self.FC3(x))
        x = F.sigmoid(self.FC4(x))
        x = F.sigmoid(self.FC5(x))
        x = F.sigmoid(self.FC6(x))
        return x