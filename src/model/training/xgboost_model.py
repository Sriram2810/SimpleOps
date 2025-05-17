# ========== XGBoost with GPU ==========
import joblib
import argparse
import pandas as pd
from typing import Text
import xgboost as xgb
import yaml
import numpy as np

def xgboost_model(config_path:Text)-> None:

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
    
    xgb_model = xgb.XGBRegressor(
        tree_method=config['train']['XGBoost']['tree_method'],
        device = config['train']['XGBoost']['device'],
        n_estimators=config['train']['XGBoost']['n_estimators'],
        max_depth=config['train']['XGBoost']['max_depth'],
        learning_rate=config['train']['XGBoost']['learning_rate'],
        subsample=config['train']['XGBoost']['subsample'],
        colsample_bytree=config['train']['XGBoost']['colsample_bytree'],
        objective=config['train']['XGBoost']['objective'],
        random_state=config['train']['XGBoost']['random_state'],
        early_stopping_rounds=config['train']['XGBoost']['early_stopping_rounds']
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save as pickle
    joblib.dump(xgb_model, config['train']['XGBoost']['model_path'])  

    print('XGBoost model has been trained!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()
    
    xgboost_model(config_path=args.config)