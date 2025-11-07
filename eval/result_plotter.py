import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(y_real: np.ndarray, y_pred: np.ndarray, model_name:str, freq: str, save_plots: bool = True, save_path: str = "prediction_plots/", show_plots: bool = True) -> None:
    """
    Plots and optionally saves a comparison of real vs predicted values for a single prediction sample.

    This function visualizes the predicted and actual values over the prediction horizon using a line plot.
    Optionally, the generated plot can be saved to disk and/or displayed on screen.

    Parameters
    ----------
    y_real : np.ndarray
        Array of real target values with shape (prediction_length,).
        Represents the true values across the prediction horizon.

    y_pred : np.ndarray
        Array of predicted values with shape (prediction_length,).
        Represents the model's forecast across the prediction horizon.

    model_name : str
        Name of the model, used in the plot title and in the saved filename.

    freq : str
        Label describing the frequency of the prediction horizon (e.g., "h" for hourly, "15min" for 15-minute steps).
        Displayed on the X-axis label.

    save_plots : bool, optional, default=True
        If True, saves the generated plot to disk.

    save_path : str, optional, default="prediction_plots/"
        Directory where the plot will be saved if `save_plots` is True.
        The directory is created automatically if it does not exist.

    show_plots : bool, optional, default=True
        If True, displays the generated plot on screen.

    Returns
    -------
    None
        The function produces a plot (and may save it) but does not return any value.

    Notes
    -----
    - The saved file is stored under the path: `<save_path>/<model_name>_prediction.png`.
    - `plt.close()` is not needed here because only one plot is generated. Add it if used inside a loop.
    """

    if save_plots:
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(y_real, label='Real', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title(f'Prediction - {model_name}')
    plt.xlabel(f'Prediction Horizon [{freq}]')
    plt.ylabel('Load [MW]')
    plt.legend()
    plt.tight_layout()

    if save_plots:
        file_path = os.path.join(save_path, f'{model_name}_prediction.png')
        plt.savefig(file_path)

    if show_plots:
        plt.show()
