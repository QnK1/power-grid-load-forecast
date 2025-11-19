from models.helpers import create_sequences
from utils.load_data import load_training_data, DataLoadingParams
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error

FEATURE_COLUMNS = [
    'load',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
]

def build_model(units: int, sequence_length: int = 24, prediction_length:int = 24, features_count: int = 7) -> keras.Sequential:
    """
    Builds a keras.Sequential model ready for training.

    Parameters
    ---------
    units: number of neurons in RNN
    sequence_length: the length of the input time window
    features_count: the number of features
    prediction_length: length of model prediction per sequence


    Returns
    ---------
    keras.Sequential
    a compiled keras.Sequential model ready for training.
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, features_count)),
        layers.SimpleRNN(units, activation='relu'),
        layers.Dense(prediction_length)
    ])

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mse', 'mape'])
    return model

# def prepare_sequences(df: pd.DataFrame, sequence_length: int, prediction_length:int, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Creates sequences from DataFrame for RNN training
#     Parameters
#     ---------
#     df: DataFrame from which to create sequences
#     sequence_length: the length of the sequences to create
#     prediction_length: length of predictions
#     features: list of features for training the model
#     label: target feature
#
#     Returns
#     ---------
#     X: np.ndarray
#        Numpy array of shape (num_samples, sequence_length, num_features)
#     y: np.ndarray
#        Numpy array of shape (num_samples, prediction_length)
#     """
#     X_data = df[features]
#     y_data = df[label]
#
#     X, y = [], []
#     for i in range(len(df) - sequence_length-prediction_length):
#         X.append(X_data.iloc[i:i+sequence_length].values)
#         y.append(y_data.iloc[i+sequence_length: i+sequence_length+prediction_length].values)
#     return np.array(X), np.array(y)


def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, epochs: int, units:int, label: str = 'load',freq: str = "1h"):
    """
    Trains the keras.Sequential model using time-series data

    Parameters
    ---------
    model: compiled keras.Sequential ready for training
    sequence_length: the length of the input time window
    epochs: number of training epochs
    prediction_length : number of timesteps into the future the model predicts


    Returns
    ---------
    None
    """
    parameters = DataLoadingParams()
    parameters.freq = freq
    parameters.interpolate_empty_values = True
    parameters.shuffle = False
    parameters.include_timeindex = True

    data, raw = load_training_data(parameters)
    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS, label)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,
        callbacks=early_stop,
        batch_size=32,
        verbose=1
    )
    file_name = f"rnn_model_{units}_{epochs}.keras"
    model_path = model_folder / file_name
    model.save(model_path)
    return history
