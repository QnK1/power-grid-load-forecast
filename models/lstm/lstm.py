import sklearn.utils
from keras.src.optimizers import Adam

from utils.load_data import load_training_data, DataLoadingParams
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import History
import pandas as pd
import numpy as np
from pathlib import Path

# This list defines the 17 features the model expects.
FEATURE_COLUMNS = [
    'load',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'timestamp_day'
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

    if len(lstm_layers) == 0:
        raise ValueError("There should be at least one LSTM layer in a LSTM Network")

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

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mse'])
    return model

def create_sequences(data: pd.DataFrame, sequence_length: int, prediction_length: int, features: list[str], label: str, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
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

    shuffle : bool, optional (default=True)
        Whether to randomly shuffle the generated (X, y) sequences. Shuffling ensures
        that the order of samples is randomized, which is appropriate for training,
        but should be set to False when preserving temporal order is important
        (e.g. for test datasets).

    Returns
    -------
    X : np.ndarray
        Numpy array of shape (num_samples, sequence_length, num_features),
        representing input sequences for the model.

    y : np.ndarray
        Numpy array of shape (num_samples, prediction_length),
        representing target values for each sequence.
    """

    if len(data) - sequence_length - prediction_length + 1 <= 0:
        raise ValueError(
            f'Not enough data to create sequences with sequence_length={sequence_length}'
            f'and prediction_length={prediction_length}.'
            f'Data length is {len(data)}.'
        )

    X_data = data[features]
    y_data = data[label]

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        X.append(X_data.iloc[i:i + sequence_length].values)
        y.append(y_data.iloc[i + sequence_length:i + sequence_length + prediction_length].values)

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        X, y = sklearn.utils.shuffle(X, y)
    return X, y

def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, predicted_label: str, epochs: int, freq: str = "1h", file_name: str = "lstm_model", early_stopping: bool = True) -> History:
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

   file_name : str, optional (default="lstm_model")
       Name of the file under which the trained model will be saved.

   early_stopping : bool, optional (default=True)
       If True, uses EarlyStopping to monitor validation loss and stop training when
       there is no improvement. Prevents overfitting and restores the best model weights.

    Returns
    -------
    History
        A Keras History object containing details about the training process,
        including loss and metrics values for each epoch.
   """

    params = DataLoadingParams()
    params.freq = freq
    params.interpolate_empty_values = True
    params.shuffle = False
    params.include_timeindex = True

    data, raw_data = load_training_data(params)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS, predicted_label)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split = 0.1 if early_stopping else None,
        callbacks = [early_stop] if early_stopping else [],
        verbose=1)

    model_path = model_folder / f"{file_name}.keras"
    model.save(model_path)

    return history

def get_model_name(lstm_layers: list[int], dense_layers: list[int], sequence_length: int, prediction_length: int, epochs: int, freq: str) -> str:
    """
    Generates a standardized model name string based on model architecture and training configuration.

    This naming convention ensures models can be easily identified when saved to disk.

    Parameters
    ----------
    lstm_layers : list[int]
        List of neuron counts for each LSTM layer, in order of appearance.
        Example: [64, 32] will be represented as "LSTM_64-32".

    dense_layers : list[int]
        List of neuron counts for each Dense layer used after LSTM layers (excluding the output layer).
        If empty, "DENSE_NONE" will be used in the name.

    sequence_length : int
        Number of past timesteps used as model input (also known as window size).

    prediction_length : int
        Number of future timesteps the model predicts.

    epochs : int
        Number of training epochs used for this model configuration.

    Returns
    -------
    str
        A descriptive model name, for example:
        "LSTM_64-32_DENSE_16-8_24_6_50"
        meaning:
        LSTM layers: [64, 32], Dense layers: [16, 8], seq_len=24, pred_len=6, epochs=50
    """

    return f"LSTM_{'-'.join(map(str, lstm_layers))}_DENSE_{'-'.join(map(str, dense_layers))}_{sequence_length}_{prediction_length}_{epochs}_{freq}"
