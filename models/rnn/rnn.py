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
label = 'load'

params = DataLoadingParams()
data, raw_data = load_training_data(params)
data = data.iloc[0:800]

def build_model(units: int, sequence_length: int = 24, features_count: int = 7) -> keras.Sequential:
    """
    Builds a keras.Sequential model ready for training.

    Parameters
    ---------
    units: number of neurons in RNN
    sequence_length: the length of the input time window
    features_count: the number of features

    Returns
    ---------
    keras.Sequential
    a compiled keras.Sequential model ready for training.

    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, features_count)),
        layers.SimpleRNN(units, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mse', 'mape'])
    return model

def prepare_sequences(df: pd.DataFrame, sequence_length: int, prediction_length:int, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences from DataFrame for RNN training
    Parameters
    ---------
    df: DataFrame from which to create sequences
    sequence_length: the length of the sequences to create
    prediction_length: length of predictions
    features: list of features for training the model
    label: target feature

    Returns
    ---------
    X: np.ndarray
       Numpy array of shape (num_samples, sequence_length, num_features)
    y: np.ndarray
       Numpy array of shape (num_samples, prediction_length)
    """
    X_data = df[features]
    y_data = df[label]

    X, y = [], []
    for i in range(len(df) - sequence_length-prediction_length):
        X.append(X_data.iloc[i:i+sequence_length].values)
        y.append(y_data.iloc[i+sequence_length: i+sequence_length+prediction_length].values)
    return np.array(X), np.array(y)

def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, epochs:int, label: str = 'load', freq: str = "1h") -> None:
    """
    Trains the keras.Sequential model using time-series data

    Parameters
    ---------
    model: compiled keras.Sequential ready for training
    sequence_length: the length of the input time window
    epochs: number of training epochs

    Returns
    ---------
    None
    """
    parameters = DataLoadingParams()
    parameters.freq = freq
    parameters.interpolate_empty_values = True
    parameters.shuffle = False

    df, raw = load_training_data(parameters)
    df = df.iloc[0:800]
    X_train, y_train = prepare_sequences(df, sequence_length, prediction_length, FEATURE_COLUMNS, label)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
    )

    current_folder = Path(__file__).parent
    model_folder = current_folder / "models"
    model_folder.mkdir(exist_ok=True)

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,
        callbacks=early_stop,
        batch_size=32,
        verbose=1
    )
    file_name = "rnn_model.keras"
    model_path = model_folder / file_name
    model.save(model_path)

def model_predict(trained_model: keras.Sequential, sequence_length:int):
    """
    Calculates predictions and mape of a trained model energy load values based on time-series data.

    Parameters
    ---------
    trained_model: trained keras.Sequential model
    sequence_length: the length of the input time window

    Returns
    ---------
    y_pred: predicted load values,
    y_true: true load values
    mape: Mean Absolute Percentage Error of the predicted values
    """

    X_data, y_true = prepare_sequences(data, sequence_length, 1, FEATURE_COLUMNS, 'load')
    y_pred = trained_model.predict(X_data)

    mape = mean_absolute_percentage_error(y_pred, y_true)
    return y_pred, y_true, mape
