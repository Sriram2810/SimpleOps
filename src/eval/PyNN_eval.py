import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import pandas as pd
import yaml
import argparse
from pathlib import Path
import json

# 1. Define the Dataset class (must match training)
class TaxiDataset(Dataset):
    def __init__(self, X, y, categorical_cols, numerical_cols):
        self.X_cat = X[categorical_cols].values.astype(np.int64)
        self.X_num = X[numerical_cols].values.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

# 2. Define the model class (must match training)
class TaxiNet(nn.Module):
    def __init__(self, emb_dims, n_num):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        emb_dim_sum = sum([y for x, y in emb_dims])
        self.fc = nn.Sequential(
            nn.Linear(emb_dim_sum + n_num, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x_cat, x_num):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = torch.cat([x, x_num], 1)
        return self.fc(x).squeeze(1)

# 3. The evaluation function
def PyNN_eval(config_path)-> None:

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config['train']['PyNN']['model_path']
    df_val = pd.read_csv(config['data']['val_data_csv'])
    categorical_cols = config['features']['categorical']
    numerical_cols = config['features']['numerical']
    batch_size = config['train']['PyNN']['batch_size']

    cat_dims = config['train']['PyNN']['cat_dims']
    emb_dims = [(x, min(50, (x+1)//2)) for x in cat_dims]

    # Prepare validation dataset and loader
    X_val = df_val[categorical_cols + numerical_cols]
    y_val = df_val['trip_duration'].values.astype(np.float32)
    val_ds = TaxiDataset(X_val, y_val, categorical_cols, numerical_cols)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TaxiNet(emb_dims, len(numerical_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run evaluation
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_cat, x_num, y_batch in tqdm(val_loader, desc="Evaluating"):
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            preds = model(x_cat, x_num).cpu().numpy()
            val_preds.append(preds)
            val_targets.append(y_batch.numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Remove NaNs
    mask = ~np.isnan(val_preds) & ~np.isnan(val_targets)
    val_preds = val_preds[mask]
    val_targets = val_targets[mask]

    rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    r2 = r2_score(val_targets, val_preds)
    print(f"PyTorch NN - RMSE: {rmse:.2f}, R²: {r2:.4f}")

    reports_folder = Path(config['eval']['reports_dir'])
    metrics_path = reports_folder/config['eval']['metrics_file_pynn']

    reports_folder.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        'PyNN': {
            "rmse": float(rmse), 
            "r2": float(r2)
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    PyNN_eval(config_path=args.config)
