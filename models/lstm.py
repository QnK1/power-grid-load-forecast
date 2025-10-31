from utils.load_data import load_training_data, DataLoadingParams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsolutePercentageError
import pandas as pd

# This list defines the 8 features the model expects.
FEATURE_COLUMNS = [
    'load',
    'temperature',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def get_model(lstm_layers: list[int], dense_layers: list[int], sequence_length: int, features_count: int = 8, dropout: float = 0.1, recurrent_dropout: float = 0.1) -> keras.Sequential:
    """
    Builds and compiles an LSTM-based neural network for time series forecasting.

    Parameters
    ----------
    lstm_layers : list[int]
        A list specifying the number of units in each LSTM layer.
        Example: [64, 32] creates two LSTM layers with 64 and 32 units respectively.

    dense_layers : list[int]
        A list specifying the number of units in each Dense layer that follows the
        LSTM layers.
        Example: [16, 8] adds two Dense layers with 16 and 8 neurons respectively.

    sequence_length : int
        The length of the input time window (number of timesteps in each sequence).

    features_count : int
        Number of features (input variables) per timestep.

    dropout : float, optional (default=0.1)
        Fraction of input units to drop for regularization in each LSTM layer.

    recurrent_dropout : float, optional (default=0.1)
        Fraction of recurrent units to drop for regularization in each LSTM layer.

    Returns
    -------
    model : keras.Sequential
        A compiled Keras Sequential model ready for training.
    """

    model = keras.Sequential()
    for i, layer in enumerate(lstm_layers):
        is_first = (i == 0)
        is_last = (i == len(lstm_layers) - 1)

        model.add(layers.LSTM(
            layer,
            input_shape=(sequence_length, features_count) if is_first else None,
            return_sequences=not is_last,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        ))

    for layer in dense_layers:
        model.add(layers.Dense(layer, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=['mae', 'mape'])
    return model


def train_models(hidden_layers: list[list[int]], epochs: list[int], freq: str = "1h"):
    params = DataLoadingParams()
    data, _ = load_training_data(params)
    print(data.columns)