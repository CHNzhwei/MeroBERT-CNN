import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import copy
from itertools import combinations
from MeroBERTCNN import MeroBERTCNN



X_train_df = pd.read_csv("../data/X_train.csv", index_col=0)
X_val_df   = pd.read_csv("../data/X_val.csv", index_col=0)
X_train = X_train_df.iloc[:, :-1].values
X_val   = X_val_df.iloc[:, :-1].values
y_train = X_train_df.iloc[:, -1].values   # Cmin
y_val   = X_val_df.iloc[:, -1].values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "LightGBM": LGBMRegressor(
        n_estimators=200, verbosity=-1
    )
}


results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    results.append({
        "Model": name,
        "Train MSE": train_mse,
        "Val MSE": val_mse,
        "Train R²": train_r2,
        "Val R²": val_r2
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Val R²", ascending=False)
print(df_results)

val_predictions = {}
for name, model in models.items():
    y_val_pred = model.predict(X_val_scaled)
    val_predictions[name] = y_val_pred
df_val_pred = pd.DataFrame(val_predictions)
df_val_pred["True_Y"] = y_val.values if hasattr(y_val, "values") else y_val
df_val_pred.to_csv("../data//ml_val_result.csv")