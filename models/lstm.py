from pathlib import Path

from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.python.layers.core import dropout

from utils.load_data import load_training_data, decode_ml_outputs, DataLoadingParams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
import numpy as np

# This list defines the 8 features the model expects.
FEATURE_COLUMNS = [
    'load',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def get_model(lstm_layers: list[int], dense_layers: list[int], sequence_length: int, prediction_length: int, features_count: int, dropout: float = 0.1, recurrent_dropout: float = 0.1) -> keras.Sequential:
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
    model.add(layers.Input(shape=(sequence_length, features_count)))
    for i, layer in enumerate(lstm_layers):
        is_last = (i == len(lstm_layers) - 1)

        model.add(layers.LSTM(
            layer,
            return_sequences=not is_last,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        ))

    for layer in dense_layers:
        model.add(layers.Dense(layer, activation='relu'))

    model.add(layers.Dense(prediction_length))

    optimizer = Adam(learning_rate=0.003)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mse'])
    return model

def create_sequences(data: pd.DataFrame, sequence_length: int, prediction_length: int, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    data = data[features]

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_length):
        X.append(data[features].iloc[i:i + sequence_length].values)
        y.append(data['load'].iloc[i + sequence_length:i + sequence_length + prediction_length].values)
    #print(y[:1000])
    return np.array(X), np.array(y)

def train_model(epochs: int, freq: str = "1h"):
    params = DataLoadingParams()
    params.freq = freq
    params.interpolate_empty_values = True

    data, raw_data = load_training_data(params)
    #print(data.columns)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    sequence_length = 24
    prediction_length = 1

    model = get_model([8], [8], sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0, recurrent_dropout=0)

    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS)
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

train_model(20)