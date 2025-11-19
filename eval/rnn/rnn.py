from matplotlib import pyplot as plt

from models.rnn.rnn import FEATURE_COLUMNS
from utils.load_data import load_test_data, DataLoadingParams, decode_ml_outputs, load_training_data
from keras.models import load_model
from pathlib import Path
from models.helpers import create_sequences
import numpy as np


def model_evaluate(sequence_length:int, prediction_length:int,file_name:str="rnn_model", freq:str='1h'):
    """
    Calculates predictions and mape of a trained model energy load values based on time-series data.

    Parameters
    ---------
    sequence_length: the length of the input time window
    prediction_length: the length of the model predictions for each sequence

    Returns
    ---------
    y_pred: predicted load values,
    y_true: true load values
    """
    main_folder = Path(__file__).parent.parent.parent
    model_path = main_folder / "models" / "rnn" / "models" / f"{file_name}.keras"
    model = load_model(model_path)

    params = DataLoadingParams()
    params.freq = freq
    params.shuffle = False
    params.include_timeindex = True

    test_data, _ = load_test_data(params)
    _, raw = load_training_data(params)

    X_test, y_test = create_sequences(test_data, sequence_length, prediction_length, FEATURE_COLUMNS, 'load', shuffle=False)
    predictions = model.predict(X_test)
    y_true = decode_ml_outputs(y_test, raw)
    y_pred = decode_ml_outputs(predictions, raw)

    mape_per_step = []

    for step in range(y_true.shape[1]):
        real_step = y_true[:, step]
        pred_step = y_pred[:, step]

        mask = real_step != 0

        if np.any(mask):
            mape = np.mean(np.abs((real_step[mask] - pred_step[mask]) / real_step[mask])) * 100
        else:
            mape = np.nan

        mape_per_step.append(mape)

    mape_per_step = np.array(mape_per_step, dtype=float)
    mape_total = np.nanmean(mape_per_step)

    with open(f'{Path(__file__).parent}/results/eval_results_{file_name}.txt', 'w') as f:
        f.write(f'step, mape_{freq}\n')
        for step, mape_step in enumerate(mape_per_step):
            f.write(f'{step}, {mape_step}\n')
        f.write(f'total, {mape_total}\n')

    return y_pred, y_true, mape_per_step