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

# This list defines the 7 features the model expects.
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

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mse'])
    return model

def create_sequences(data: pd.DataFrame, sequence_length: int, prediction_length: int, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a time-series DataFrame into sliding window sequences for supervised learning.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with all features already processed and scaled.

    sequence_length : int
        Number of historical timesteps used as input (window size).

    prediction_length : int
        Number of future timesteps the model should predict.

    features : list[str]
        List of feature column names to be used as model input.

    label : str
        Column name containing the target values to predict.

    Returns
    -------
    X : np.ndarray
        Numpy array of shape (num_samples, sequence_length, num_features),
        representing input sequences for the model.

    y : np.ndarray
        Numpy array of shape (num_samples, prediction_length),
        representing target values for each sequence.
    """

    x_data = data[features]
    y_data = data[label]

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_length):
        X.append(x_data.iloc[i:i + sequence_length].values)
        y.append(y_data.iloc[i + sequence_length:i + sequence_length + prediction_length].values)
    return np.array(X), np.array(y)

def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, predicted_label: str, epochs: int, freq: str = "1h") -> None:
    """
   Trains the provided model using historical time-series data.

   Parameters
   ----------
   model : keras.Sequential
       Compiled Keras model created via `get_model`, ready for training.

   sequence_length : int
       Length of input sequences passed to the model.

   prediction_length : int
       Number of timesteps into the future the model predicts.

   predicted_label : str
       Column name containing the target values to be predicted.

   epochs : int
       Number of training epochs.

   freq : str, optional (default="1h")
       Data frequency used for training.

   Returns
   -------
   None
       The model is trained and stored in the "models" folder.
   """

    params = DataLoadingParams()
    params.freq = freq
    params.interpolate_empty_values = True
    params.shuffle = False

    data, raw_data = load_training_data(params)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS, predicted_label)
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

# Training section (to be removed later)

sequence_length = 24
prediction_length = 1
model = get_model([8], [8], sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0, recurrent_dropout=0)
predicted_label = 'load'

epochs = 50

train_model(model, sequence_length, prediction_length, predicted_label, epochs)