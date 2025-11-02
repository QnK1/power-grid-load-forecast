from utils.load_data import load_test_data, DataLoadingParams, decode_ml_outputs
from models.lstm.lstm import create_sequences
import numpy as np
from keras.models import load_model
from pathlib import Path

FEATURE_COLUMNS = [
    'load',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def evaluate_models(sequence_length: int, prediction_length: int, file_name: str, freq: str = '1h'):
    project_folder = Path(__file__).parent.parent.parent
    model_path = project_folder / 'models' / 'lstm' / 'models' / f'{file_name}.keras'
    model = load_model(model_path)

    params = DataLoadingParams()

    test_data, raw_data = load_test_data(params)
    X_test, y_test = create_sequences(test_data, sequence_length, prediction_length, FEATURE_COLUMNS, 'load')

    predictions = model.predict(X_test)
    y_real = decode_ml_outputs(y_test, raw_data)
    y_pred = decode_ml_outputs(predictions, raw_data)

    mape_per_step = np.mean(np.abs((y_real - y_pred) / y_real), axis=0) * 100
    mape_total = np.mean(mape_per_step)

    with open(f'{Path(__file__).parent}/results/eval_results_{file_name}.txt', 'w') as f:
        f.write(f'step, mape_{freq}\n')
        for step, mape_step in enumerate(mape_per_step):
            f.write(f'{step}, {mape_step}\n')
        f.write(f'total, {mape_total}\n')


evaluate_models(24, 3, 'LSTM_8_DENSE_8_24_3_10')