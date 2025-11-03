import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(y_real: np.ndarray, y_pred: np.ndarray, examples: list[int], model_name:str, save_plots: bool = True, save_path: str = "prediction_plots/", show_plots: bool = True) -> None:
    """
    Plot and optionally save prediction vs real values for selected examples.

    Parameters
    ----------
    y_real : np.ndarray
        Array of real target values with shape (num_samples, prediction_length).
        Values from shape[1] are plotted.
    y_pred : np.ndarray
        Array of predicted values with shape (num_samples, prediction_length).
        Values from shape[1] are plotted.
    examples : list[int]
        List of sample indices to visualize. Each index corresponds to one row in y_real/y_pred (shape[0]).
    model_name : str
        Name of the model, used in plot titles and saved filenames.
    save_plots : bool, default=True
        If True, saves generated plots to disk.
    save_path : str, default="prediction_plots/"
        Directory where plots will be saved if `save_plots` is True. Created automatically if it doesn't exist.
    show_plots : bool, default=True
        If True, displays generated plots on screen.

    Returns
    -------
    None
        The function produces plots and may save them to files but does not return any value.

    Notes
    -----
    - Each selected example generates a separate plot.
    - Files are saved in the format: `<model_name>_prediction_<index>.png`
    - `plt.close()` is used to avoid memory issues when plotting in loops.
    """

    if save_plots:
        os.makedirs(save_path, exist_ok=True)

    for example in examples:
        plt.figure(figsize=(8, 4))
        plt.plot(y_real[example], label='Real', color='blue')
        plt.plot(y_pred[example], label='Predicted', color='red')
        plt.title(f'Prediction Example {example} - {model_name}')
        plt.xlabel('Prediction Horizon')
        plt.ylabel('Load')
        plt.legend()
        plt.tight_layout()

        if save_plots:
            file_path = os.path.join(save_path, f'{model_name}_prediction_{example}.png')
            plt.savefig(file_path)

        if show_plots:
            plt.show()

        plt.close()