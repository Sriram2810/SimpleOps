import pandas as pd
import argparse
import yaml
from typing import Text
from sklearn.model_selection import train_test_split

def data_split(config_path: Text) -> None:
    """
    Load data, split into train/val, validate features, save CSVs, and update params.yaml with actual features.
    """
    # Load config
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    df = pd.read_csv(config['data']['processed_data_csv'])

    # Train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=config['data']['test_size'],
        random_state=config['base']['random_seed']
    )

    # Save without index
    train_df.to_csv(config['data']['train_data_csv'], index=False)
    val_df.to_csv(config['data']['val_data_csv'], index=False)
    print("✅ Train and validation data saved.")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
