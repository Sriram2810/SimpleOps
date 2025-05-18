import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from typing import Text
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example: set the seed to 42 (or any integer you like)
set_seed(42)

class TaxiDataset(Dataset):
    def __init__(self, X, y, categorical_cols, numerical_cols):
        self.X_cat = X[categorical_cols].values.astype(np.int64)
        self.X_num = X[numerical_cols].values.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

class TaxiNet(nn.Module):
    def __init__(self, emb_dims, n_num):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        emb_dim_sum = sum([y for _, y in emb_dims])
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

def PyNN(config_path: Text) -> None:

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load data
    df_t = pd.read_csv(config['data']['train_data_csv']) 
    df_v = pd.read_csv(config['data']['val_data_csv'])

    categorical_cols = config['features']['categorical']
    numerical_cols = config['features']['numerical']

    X_train = df_t[categorical_cols + numerical_cols]
    y_train = df_t['trip_duration'].values.astype(np.float32)
    X_val = df_v[categorical_cols + numerical_cols]
    y_val = df_v['trip_duration'].values.astype(np.float32)

    train_ds = TaxiDataset(X_train, y_train, categorical_cols, numerical_cols)
    val_ds = TaxiDataset(X_val, y_val, categorical_cols, numerical_cols)

    BATCH_SIZE = config['train']['PyNN']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=config['train']['PyNN']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=config['train']['PyNN']['num_workers'])

    cat_dims = [int(df_t[col].nunique()) for col in categorical_cols]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaxiNet(emb_dims, len(numerical_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['PyNN']['lr'])
    criterion = nn.MSELoss()

    EPOCHS = config['train']['PyNN']['epochs']
    patience = config['train']['PyNN']['patience']
    patience_counter = config['train']['PyNN']['patience_counter']
    best_r2 = -np.inf

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x_cat, x_num, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x_cat, x_num, y_batch = x_cat.to(device), x_num.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_cat, x_num)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cat, x_num, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x_cat, x_num = x_cat.to(device), x_num.to(device)
                preds = model(x_cat, x_num).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        print(f"Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.4f} | Val RMSE: {val_rmse:.2f} | Val R2: {val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), config['train']['PyNN']['model_path'])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed.")
    config['train']['PyNN']['cat_dims'] = cat_dims
    with open('params.yaml', 'w') as f:
        yaml.safe_dump(config, f)

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()
    
    PyNN(config_path=args.config)
