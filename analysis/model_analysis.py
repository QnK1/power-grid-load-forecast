from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np
import os

class ModelPlotCreator:
    def __init__(self):
        """
        Utility class for generating and saving visualizations related to model performance.

        This class provides methods for plotting:
        - Real vs predicted values for a single forecast sample.
        - Training and validation loss curves across epochs (learning curves).

        Plots can be displayed on screen, saved to organized subfolders, or both.
        The class automatically constructs the base directory for saving plots relative
        to the file location to ensure a consistent and portable folder structure.

        Attributes
        ----------
        save_path : str
            Base directory path where all model-related plots will be stored.
            Subfolders for specific plot types (e.g., prediction plots, learning curves)
            are created within this path.

        Notes
        -----
        - The class assumes that provided data arrays or history objects come from an already
          trained model.
        - Subfolders are automatically created when saving plots, if they do not exist.
        """

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(base_path, 'plots/model_plots/')

    def plot_predictions(self, y_real: np.ndarray, y_pred: np.ndarray, model_name: str, folder: str, freq: str, save_plot: bool = True, show_plot: bool = True) -> None:
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

        folder : str
            Subfolder (relative to the class-level save_path) where the plot will be stored.
            Useful for organizing plots by model, dataset split, or experiment series.
            Example: "lstm", "rnn", "committee"

        freq : str
            Label describing the frequency of the prediction horizon (e.g., "h" for hourly, "15min" for 15-minute steps).
            Displayed on the X-axis label.

        save_plot : bool, optional, default=True
            If True, saves the generated plot to disk.

        show_plot : bool, optional, default=True
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

        save_path = os.path.join(self.save_path, 'prediction_plots', folder)

        if save_plot:
            os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(y_real, label='Real', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.title(f'Prediction - {model_name}')
        plt.xlabel(f'Prediction Horizon [{freq}]')
        plt.ylabel('Load [MW]')
        plt.legend()
        plt.tight_layout()

        if save_plot:
            file_path = os.path.join(save_path, f'{model_name}_prediction.png')
            plt.savefig(file_path)

        if show_plot:
            plt.show()

        plt.close()

    def plot_learning_curves(self, history: History, model_name: str, folder: str, save_plot: bool = True, show_plot: bool = True) -> None:
        """
        Plots the training and validation loss curves across epochs for a trained model.

        This function visualizes how the model's loss evolved during training, enabling detection
        of issues such as overfitting or underfitting. The plot shows both the training and
        validation loss values recorded at each epoch. The generated plot can be displayed
        on screen and/or saved to disk.

        Parameters
        ----------
        history : keras.callbacks.History
            History object returned by the `model.fit()` method. Must contain both
            `'loss'` and `'val_loss'` entries.

        model_name : str
            Name of the model, used in the plot title and saved filename.

        folder : str
            Subfolder name where the plot will be saved relative to the base save path.
            Example: `"lstm"`, `"mlp"`.

        save_plot : bool, optional, default=True
            If True, saves the generated plot to disk.

        show_plot : bool, optional, default=True
            If True, displays the generated plot on screen.

        Returns
        -------
        None
            The function produces plots (and may save them), but does not return any value.

        Notes
        -----
        - The saved file follows the naming convention: `<model_name>_learning_curve.png`
        - Raises a `ValueError` if the provided `history` object does not contain validation loss data.
        """

        save_path = os.path.join(self.save_path, 'learning_curves', folder)

        if history.history.get('val_loss') is None:
            raise ValueError('You have to provide a history of a model that has been validated.')

        if save_plot:
            os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title(f'Learning curve - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.legend()
        plt.tight_layout()

        if save_plot:
            file_path = os.path.join(save_path, f'{model_name}_learning_curve.png')
            plt.savefig(file_path)

        if show_plot:
            plt.show()

        plt.close()

    def plot_multiple_predictions(self, y_real: np.ndarray, y_pred: list[np.ndarray], model_names: list[str], folder: str, freq: str, save_plot: bool = True, show_plot: bool = True) -> None:
        """
        Plots and optionally saves a comparison of real values versus multiple prediction outputs from different models.

        This function visualizes the true target values together with predictions from several models on a single plot,
        enabling direct comparison of model performance over the prediction horizon.

        Parameters
        ----------
        y_real : np.ndarray
            Array of real target values with shape (prediction_length,).
            Represents the ground truth across the prediction horizon.

        y_pred : list[np.ndarray]
            A list of prediction arrays, each with shape (prediction_length,).
            Each element corresponds to predictions from a different model.

        model_names : list[str]
            A list of model names, each associated with a corresponding prediction array in `y_pred`.
            Used for plot labeling and filename generation.

        folder : str
            Subfolder (relative to the class-level `save_path`) where the plot will be stored.

        freq : str
            Label describing the frequency of the prediction horizon (e.g., "h" for hourly,
            "15min" for 15-minute resolution). Displayed on the X-axis label.

        save_plot : bool, optional, default=True
            If True, saves the generated plot to disk.

        show_plot : bool, optional, default=True
            If True, displays the generated plot on screen.

        Returns
        -------
        None
            The function creates a comparison plot (and may save it), but does not return any value.

        Notes
        -----
        - The saved file is stored under the path:
          `<save_path>/prediction_plots/<folder>/<model1-model2-...-modelN>_prediction.png`.
        - The function assumes that all prediction arrays in `y_pred` share the same prediction horizon length as `y_real`.
        """

        save_path = os.path.join(self.save_path, 'prediction_plots', folder)

        if len(y_pred) != len(model_names):
            raise ValueError('There should be as many prediction arrays as there are model names.')

        if save_plot:
            os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(y_real, label='Real', color='blue')
        for pred, name in zip(y_pred, model_names):
            plt.plot(pred, label=name.capitalize())
        plt.title(f'Predictions for multiple models')
        plt.xlabel(f'Prediction Horizon [{freq}]')
        plt.ylabel('Load [MW]')
        plt.legend()
        plt.tight_layout()

        if save_plot:
            joined_names = '-'.join(model_names)
            file_path = os.path.join(save_path, f'{joined_names}_predictions.png')
            plt.savefig(file_path)

        if show_plot:
            plt.show()

        plt.close()

    def compare_average_mape(self, model_names: list[str], average_mape: list[float], folder: str, save_plot: bool = True, show_plot: bool = True) -> None:
        """
        Creates a bar chart comparing average MAPE values across models.

        The method visualizes the Mean Absolute Percentage Error (MAPE) for multiple models
        using a bar plot, allowing easy comparison of their average performance.
        Optionally, the plot can be saved to disk or displayed interactively.

        Parameters
        ----------
        model_names : list of str
            Names of the models to be compared.

        average_mape : list of float
            Corresponding average MAPE values for each model.

        folder : str
            Subfolder (relative to self.save_path/average_mape_plots) where the plot will be saved.

        save_plot : bool, optional (default=True)
            If True, saves the generated plot to disk.

        show_plot : bool, optional (default=True)
            If True, displays the plot interactively.

        Returns
        -------
        None
        """

        save_path = os.path.join(self.save_path, 'average_mape_plots', folder)

        if len(model_names) != len(average_mape):
            raise ValueError('Number of models should equal the number of average values of MAPE.')

        if save_plot:
            os.makedirs(save_path, exist_ok=True)

        plt.bar(model_names, average_mape)
        plt.xlabel('Model')
        plt.ylabel('Average MAPE')
        plt.title('Comparison of average MAPE')
        plt.xticks(rotation=60)

        plt.tight_layout()

        if save_plot:
            file_path = os.path.join(save_path, 'average_mape_comparisson.png')
            plt.savefig(file_path)

        if show_plot:
            plt.show()

        plt.close()

    def plot_mape_over_horizon(self, model_name: str, mape_values: list[float], folder: str, save_plot: bool = True, show_plot: bool = True) -> None:
        """
        Plots MAPE values across prediction horizons for a single model.

        The method visualizes how the Mean Absolute Percentage Error (MAPE)
        changes with the length of the prediction horizon. This helps assess
        model performance degradation over time. Optionally, the plot can be
        saved to disk or displayed interactively.

        Parameters
        ----------
        model_name : str
            Name of the model for which the MAPE values are plotted.

        mape_values : list of float
            Sequence of MAPE values corresponding to increasing prediction horizons.

        folder : str
            Subfolder (relative to self.save_path/mape_over_horizon_plots)
            where the plot will be saved.

        save_plot : bool, optional (default=True)
            If True, saves the generated plot to disk.

        show_plot : bool, optional (default=True)
            If True, displays the plot interactively.

        Returns
        -------
        None
        """

        save_path = os.path.join(self.save_path, 'mape_over_horizon_plots', folder)

        if save_plot:
            os.makedirs(save_path, exist_ok=True)

        plt.plot(mape_values)
        plt.xlabel('Prediction Horizon')
        plt.ylabel('MAPE')
        plt.title(f'MAPE over horizon for {model_name}')

        plt.tight_layout()

        if save_plot:
            file_path = os.path.join(save_path, f'{model_name}_mape_over_horizon.png')
            plt.savefig(file_path)

        if show_plot:
            plt.show()

        plt.close()
