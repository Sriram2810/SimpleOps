import pandas as pd
import argparse
from typing import Text
import yaml

def null_handling(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    df = pd.read_csv(config['data']['raw_data_csv'])

    for col in ['passenger_count', 'RatecodeID', 'congestion_surcharge', 'airport_fee']:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical column with mode
    for col in ['store_and_fwd_flag']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.to_csv(config['data']['null_handled_data_csv'], index=False)

    print('Null handling completed and saved to CSV file successfully.')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    null_handling(config_path=args.config)

    