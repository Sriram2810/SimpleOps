import argparse
import pandas as pd
from typing import Text
import yaml

def data_load(config_path: Text) -> None:
    """
    Load data from a parquet file and save it as a CSV file.
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    df = pd.read_parquet(config['data']['raw_data_parquet'])
    df.to_csv(config['data']['raw_data_csv'], index=False)

    print('CSV file saved successfully.')

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config',required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)