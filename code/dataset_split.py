import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import random

df = pd.read_csv("../data/Meropenem_samples.csv", index_col=0)
df = df.dropna(subset=["C"])
target_col = df.columns[-1]
target = df[target_col]

# 2. one-hot "Frequency" and "Drug dose"
df_X = df.drop(columns=[target_col])
df_X = pd.get_dummies(df_X, columns=["Frequency"],dtype=int)
df_X = pd.get_dummies(df_X, columns=["Drug dose"],dtype=int)
df_X = df_X.fillna(df.mean(numeric_only=True))

df_processed = (df_X.fillna(lambda x: x.mean() if x.dtype.kind in "biufc" else x)
        .pipe(lambda d: d.assign(**{c: (d[c]-d[c].min())/(d[c].max()-d[c].min()) 
                                    for c in d.select_dtypes(include="number").columns})))

df_processed = pd.concat([df_X, target], axis=1)

X_36 = df_processed.iloc[:, :-1].values
y = df_processed.iloc[:, -1].values   # sample Cmin
X_768 = np.load("./biobert_embeddings.npy") # use BioBERT_encoding.py

random_state = 666 
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
 
sorted_idx = np.argsort(y) #sorted by Cmin
X_sorted = X_36[sorted_idx]
X_768_sorted = X_768[sorted_idx]
y_sorted = y[sorted_idx]

n_samples = len(y)
test_size = 0.2
n_val = int(n_samples * test_size)


val_idx = np.arange(0, n_samples, int(1 / test_size))[:n_val]
train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

X_train_raw = X_sorted[train_idx]
X_768_train_raw = X_768_sorted[train_idx]
y_train_raw = y_sorted[train_idx]

X_val_raw = X_sorted[val_idx]
X_768_val_raw = X_768_sorted[val_idx]
y_val_raw = y_sorted[val_idx]

# shuffle
rng = np.random.RandomState(random_state)
train_perm = rng.permutation(len(X_train_raw))
val_perm   = rng.permutation(len(X_val_raw))

X_train_raw = X_train_raw[train_perm]
X_768_train_raw = X_768_train_raw[train_perm]
y_train_raw = y_train_raw[train_perm]

X_val_raw = X_val_raw[val_perm]
X_768_val_raw = X_768_val_raw[val_perm]
y_val_raw = y_val_raw[val_perm]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)
# joblib.dump(scaler, "./scaler.pkl")

X_train_36 = torch.tensor(X_train_scaled, dtype=torch.float32)
X_train_768 = torch.tensor(X_768_train_raw, dtype=torch.float32)

X_val_36 = torch.tensor(X_val_scaled, dtype=torch.float32)
X_val_768 = torch.tensor(X_768_val_raw, dtype=torch.float32)

y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
y_val   = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_36 = X_train_36.to(device)
X_train_768 = X_train_768.to(device)
y_train = y_train.to(device)

X_val_36 = X_val_36.to(device)
X_val_768 = X_val_768.to(device)
y_val = y_val.to(device)