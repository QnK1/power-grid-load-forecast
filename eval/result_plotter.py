import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(y_real: np.ndarray, y_pred: np.ndarray, examples: list[int], model_name:str, save_plots: bool = True, save_path: str = "prediction_plots/", show_plots: bool = True) -> None:
    if save_plots:
        os.makedirs(save_path, exist_ok=True)

    for example in examples:
        plt.plot(y_real[example], label='Real', color='blue')
        plt.plot(y_pred[example], label='Predicted', color='red')
        plt.title(f'Prediction Example {example} - {model_name}')
        plt.xlabel('Prediction Horizon')
        plt.ylabel('Load')
        plt.legend()

        if save_plots:
            file_path = os.path.join(save_path, f'{model_name}_prediction_{example}.png')
            plt.savefig(file_path)

        if show_plots:
            plt.show()