import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_kde_plot(y_val, y_pred, model_name: str, save_path: str):
    """
    Generates and saves a KDE plot comparing actual and predicted values.

    Parameters:
    - y_val: array-like of true values
    - y_pred: array-like of predicted values
    - model_name: string, name of the ML model (used in plot title)
    - save_path: string, path to save the generated plot
    """
    df = pd.DataFrame({'y_val': y_val, 'y_pred': y_pred})

    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['y_val'], label='Actual (y_val)', fill=True, common_norm=False)
    sns.kdeplot(df['y_pred'], label='Predicted (y_pred)', fill=True, common_norm=False)
    plt.title(f'Distribution of Actual vs Predicted for {model_name} Model')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()