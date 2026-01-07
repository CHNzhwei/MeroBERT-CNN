import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score,roc_auc_score,roc_curve



class MeroBERTCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # --- 36D → 768 ---
        self.fc_proj = nn.Sequential(
            nn.Linear(36, 256),
            nn.ReLU(),
            nn.Linear(256, 768),
            nn.ReLU()
        )

        # --- 1×2×768 ---
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))  # (batch, 8, 1, 383)
        )

        self.flat_dim = 8 * 1 * 383

        # --- MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x36, x768):
        struct768 = self.fc_proj(x36)                 # (batch, 768)
        fused = torch.stack([struct768, x768], dim=1) # (batch, 2, 768)
        fused = fused.unsqueeze(1)                    # (batch, 1, 2, 768)

        x = self.conv(fused)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)                          # (batch, 2)
        return logits
    
    
def soft_constraint_loss(y_pred, y_true, lam=1.0, eps=1e-8):
    """
    lam: 0.5～3
    """
    mse = nn.MSELoss()
    base_loss = mse(y_pred, y_true)
    rel_err = torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)
    violation = torch.clamp(rel_err - 0.3, min=0.0)
    constraint_loss = violation.mean()  # 对超出部分施加惩罚
    return base_loss + lam * constraint_loss

