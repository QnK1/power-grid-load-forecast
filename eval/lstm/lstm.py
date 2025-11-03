from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.lstm.lstm import create_sequences
import numpy as np
from keras.models import load_model
from pathlib import Path

FEATURE_COLUMNS = [
    'load',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def evaluate_models(sequence_length: int, prediction_length: int, file_name: str, freq: str = '1h', save_results: bool = True):
    project_folder = Path(__file__).parent.parent.parent
    model_path = project_folder / 'models' / 'lstm' / 'models' / f'{file_name}.keras'
    model = load_model(model_path)

    params = DataLoadingParams()
    params.shuffle = False

    test_data, _ = load_test_data(params)
    _, raw_data = load_training_data(params)
    X_test, y_test = create_sequences(test_data, sequence_length, prediction_length, FEATURE_COLUMNS, 'load')

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
