import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score,roc_auc_score,roc_curve
from MeroBERTCNN import MeroBERTCNN, soft_constraint_loss


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

with torch.no_grad():
    y_pred_tensor = model(X_val_36, X_val_768)
    y_pred_np = y_pred_tensor.cpu().numpy().ravel()
val_pred_df = pd.DataFrame({
        "MeroBERT-CNN": y_pred_np,
        "True_Y": y_val
    })
val_pred_df.to_csv("../data/cnn_val_result.csv", index=False)