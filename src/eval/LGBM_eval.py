from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import argparse
import yaml
import joblib
from typing import Text
import pandas as pd
from pathlib import Path
import json


def LGBM_eval(config_path:Text)->None:

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Predict on validation set

    lgbm_model = joblib.load(config['train']['LightGBM']['model_path'])
    df_v = pd.read_csv(config['data']['val_data_csv'])

    categorical_cols = config['features']['categorical']
    numerical_cols = config['features']['numerical']

    X_val = df_v[categorical_cols + numerical_cols]
    y_val = df_v['trip_duration'].values.astype(np.float32)

    y_pred = lgbm_model.predict(X_val)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Compute R² score
    r2 = r2_score(y_val, y_pred)

    print(f"LGB_Model (GPU) - RMSE: {rmse:.2f}, R²: {r2:.4f}")

    reports_folder = Path(config['eval']['reports_dir'])
    metrics_path = reports_folder/config['eval']['metrics_file_lgbm']

    reports_folder.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        'LGBM': {
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

    LGBM_eval(config_path=args.config)