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
    def __init__(self, num_classes=2):
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

X_train_36 = ...
X_train_768 = ...
y_train = ...
X_val_36 = ...
X_val_768 = ...
y_val = ...  # from dataset_split.py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MeroBERTCNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)
criterion = soft_constraint_loss

patience = 500
best_val_r2 = -float("inf")
epochs_no_improve = 0
best_model_state = None
num_epochs = 5000

for epoch in range(num_epochs):
    # =========================
    #       Train
    # =========================
    model.train()
    optimizer.zero_grad()

    preds = model(X_train_36, X_train_768)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    # =========================
    #       Validate
    # =========================
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_36, X_val_768)
        val_r2 = r2_score(
            y_val.cpu().numpy(), 
            val_preds.cpu().numpy()
        )
        val_loss = criterion(val_preds, y_val)
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1:04d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val R²: {val_r2:.4f}"
        )

    # =========================
    #   Early Stopping
    # =========================
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(
            f"⏹️ Early stopping at epoch {epoch+1} "
            f"(no R² improvement for {patience} epochs)"
        )
        break

# =========================
#   Load best model
# =========================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✅ Loaded best model with Val R² = {best_val_r2:.4f}")

# torch.save(model.state_dict(), "./model_save/model_weights_fused_model.pth")