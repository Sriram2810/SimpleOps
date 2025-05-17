from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd
import yaml
import argparse
import torch
from typing import Text
import numpy as np
import joblib

def Tabnet_model(config_path:Text)-> None:

    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)

        
    df_t = pd.read_csv(config['data']['train_data_csv']) 
    df_v = pd.read_csv(config['data']['val_data_csv'])
    
    categorical_cols = config['features']['categorical']
    numerical_cols = config['features']['numerical']
    
    X_train = df_t[categorical_cols + numerical_cols]
    y_train = df_t['trip_duration'].values.astype(np.float32)
    X_val = df_v[categorical_cols + numerical_cols]
    y_val = df_v['trip_duration'].values.astype(np.float32)

    tabnet = TabNetRegressor(device_name='cuda' if torch.cuda.is_available() else 'cpu')
    tabnet.fit(
        X_train.values, y_train.reshape(-1, 1),
        eval_set=[(X_val.values, y_val.reshape(-1, 1))],
        eval_metric=config['train']['TabNet']['eval_metric'],
        max_epochs=config['train']['TabNet']['max_epochs'],
        patience=config['train']['TabNet']['patience'],
        batch_size=config['train']['TabNet']['batch_size'], 
        virtual_batch_size=config['train']['TabNet']['virtual_batch_size'],
        num_workers=config['train']['TabNet']['num_workers']
    )

    joblib.dump(tabnet, config['train']['TabNet']['model_path'])  

    print('TabNet model has been trained!')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    Tabnet_model(config_path=args.config)