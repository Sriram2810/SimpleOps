import pandas as pd
import argparse
from typing import Text
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler

def time_of_day(hour):
    if hour < 6: return 'night'
    elif hour < 12: return 'morning'
    elif hour < 18: return 'afternoon'
    else: return 'evening'

def featurize(config_path:Text)-> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    df = pd.read_csv(config['data']['null_handled_data_csv'], parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)]

    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['is_peak'] = ((df['pickup_hour'].between(7,9)) | (df['pickup_hour'].between(16,18))).astype(int)

    
    df['time_of_day'] = df['pickup_hour'].apply(time_of_day)

    categorical_cols = [
        'VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type',
        'PULocationID', 'DOLocationID', 'time_of_day'
    ]
    numerical_cols = [
        'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge',
        'pickup_hour', 'pickup_weekday', 'is_weekend', 'pickup_day', 'month', 'is_peak', 'manhattan_dist'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    config['features'] = config.get('features', {}) 
    config['features']['numerical'] = numerical_cols
    config['features']['categorical'] = categorical_cols

    # Label encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Standardize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    with open('params.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    df.to_csv(config['data']['processed_data_csv'],index=False)
    print('Featurization completed and saved to CSV file successfully.')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)