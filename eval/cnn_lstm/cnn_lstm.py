from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.helpers import create_sequences
import numpy as np
from keras.models import load_model
from pathlib import Path

FEATURE_COLUMNS = [
    'load',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'timestamp_day'
]

def evaluate_models(sequence_length: int, prediction_length: int, file_name: str, freq: str = '1h', save_results: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a trained CNN-LSTM model on test data and returns real and predicted
    load values for further analysis.

    The evaluation pipeline performs the following steps:
    1. Loads a saved model from disk.
    2. Loads test and raw (unscaled) datasets.
    3. Builds sliding-window sequences compatible with the model input.
    4. Generates multi-step forecasts for each test window.
    5. Decodes predictions and ground truth back to MW units.
    6. Computes per-step MAPE and total MAPE over the full prediction horizon.
    7. Optionally saves the metric results to a text file.

    Parameters
    ----------
    sequence_length : int
        Number of past timesteps used as input (sliding-window size).

    prediction_length : int
        Number of future timesteps predicted by the model.

    file_name : str
        Base name of the saved `.keras` model file.
        Must match the architecture and training configuration used during training.

    freq : str, optional (default="1h")
        Frequency of the input data used for evaluation.
        Must be consistent with the data frequency used during model training.

    save_results : bool, optional (default=True)
        If True, saves per-step and aggregated MAPE values to:
        `results/eval_results_<file_name>.txt`.

    Returns
    -------
    y_real : np.ndarray
        Array of shape (num_samples, prediction_length) containing the true load
        values in MW, decoded back from scaled form.

    y_pred : np.ndarray
        Array of shape (num_samples, prediction_length) containing the model’s
        predicted load values in MW.

    Notes
    -----
    - This function performs **direct multi-step forecasting**, meaning the model
      predicts all future steps in a single forward pass.
    - The returned arrays can be used for plotting, error decomposition, or other
      post-evaluation analytics.
    - It is crucial that `FEATURE_COLUMNS` and preprocessing steps match exactly
      those used during the training phase.
    """


    project_folder = Path(__file__).parent.parent.parent
    model_path = project_folder / 'models' / 'cnn_lstm' / 'models' / f'{file_name}.keras'
    model = load_model(model_path)

    params = DataLoadingParams()
    params.freq = freq
    params.shuffle = False
    params.include_timeindex = True

    test_data, _ = load_test_data(params)
    _, raw_data = load_training_data(params)
    X_test, y_test = create_sequences(test_data, sequence_length, prediction_length, FEATURE_COLUMNS, 'load', shuffle=False)

    predictions = model.predict(X_test)
    y_real = decode_ml_outputs(y_test, raw_data)
    y_pred = decode_ml_outputs(predictions, raw_data)

    # Calculating MAPE for each predicted step (y_real.shape[1] == prediction_length)
    mape_per_step = []

    for step in range(y_real.shape[1]):
        real_step = y_real[:, step]
        pred_step = y_pred[:, step]

        mask = real_step != 0

        if np.any(mask):
            mape = np.mean(np.abs((real_step[mask] - pred_step[mask]) / real_step[mask])) * 100
        else:
            mape = np.nan

        mape_per_step.append(mape)

    mape_per_step = np.array(mape_per_step, dtype=float)
    mape_total = np.nanmean(mape_per_step)

    if save_results:
        with open(f'{Path(__file__).parent}/results/eval_results_{file_name}.txt', 'w') as f:
            f.write(f'step, mape_{freq}\n')
            for step, mape_step in enumerate(mape_per_step):
                f.write(f'{step}, {mape_step}\n')
            f.write(f'total, {mape_total}\n')

    return y_real, y_pred
