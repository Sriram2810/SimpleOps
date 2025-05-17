import joblib
import argparse
import pandas as pd
from typing import Text
import lightgbm as lgb
import yaml
import numpy as np

def LGBM_model(config_path:Text)-> None:

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

    lgbm_model = lgb.LGBMRegressor(
        device=config['train']['LightGBM']['device'],
        num_leaves=config['train']['LightGBM']['num_leaves'],
        learning_rate=config['train']['LightGBM']['learning_rate'],
        n_estimators=config['train']['LightGBM']['n_estimators'],
        min_child_samples=config['train']['LightGBM']['min_child_samples'],
        early_stopping_rounds=config['train']['LightGBM']['early_stopping_rounds'],
        random_state=config['base']['random_seed'],
        verbose = 0
    )
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Save as pickle
    joblib.dump(lgbm_model, config['train']['LightGBM']['model_path'])  

    print('LGBM model has been trained!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()
    
    LGBM_model(config_path=args.config)