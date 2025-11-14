from pathlib import Path

from keras.src.optimizers import Adam
from models.helpers import create_sequences
from utils.load_data import load_training_data, DataLoadingParams
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import History

# This list defines the 10 features the model expects.
FEATURE_COLUMNS = [
    'load',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'timestamp_day'
]

def get_model(convolution_layers: list[int], kernel_sizes: list[int], dense_layers: list[int], sequence_length: int, prediction_length: int, features_count: int, pooling_type: str = 'mean', pool_size: int = 2, spatial_dropout: float = 0.1) -> keras.Sequential:
    """
    Builds and compiles a 1D Convolutional Neural Network (CNN) for time series forecasting.

    Parameters
    ----------
    convolution_layers : list[int]
        A list specifying the number of filters in each Conv1D layer.
        Example: [32, 16] creates two Conv1D layers with 32 and 16 filters respectively.

    kernel_sizes : list[int]
        A list specifying the kernel size (window length) for each Conv1D layer.
        Each convolution layer must have exactly one kernel size assigned.
        Example: [3, 2] sets the first Conv1D layer to look at 3 timesteps,
        the second at 2 timesteps.

    dense_layers : list[int]
        A list specifying the number of units in each Dense layer after the convolutional layers.
        Example: [16, 8] adds two Dense layers with 16 and 8 neurons respectively.

    sequence_length : int
        The number of timesteps in each input sequence.

    prediction_length : int
        The number of steps ahead the model predicts (output size).

    features_count : int
        Number of features (input variables) per timestep.

    pooling_type : str, optional (default='mean')
        Type of pooling to apply after Conv1D layers. Options:
        - 'mean' : MeanPooling1D
        - 'max' : MaxPooling1D
        - 'none': no pooling

    pool_size : int, optional (default=2)
        Size of the pooling window, ignored if pooling_type='none'.

    spatial_dropout : float, optional (default=0.1)
        Fraction of feature maps to drop for regularization after each Conv1D layer.
        If 0, dropout is not applied.

    Returns
    -------
    model : keras.Sequential
        A compiled Keras Sequential model ready for training.
        The model consists of Conv1D layers (with optional SpatialDropout1D and pooling),
        followed by Flatten and Dense layers, ending with a Dense layer for predictions.
    """

    if len(convolution_layers) == 0:
        raise ValueError('There should be at least one convolution layer in a CNN')

    if len(convolution_layers) != len(kernel_sizes):
        raise ValueError('Each convolution layer should have exactly one kernel size assigned to it.')

    if pooling_type not in ['mean', 'max', 'none']:
        raise ValueError('Available types of pooling are: mean, max, none.')

    model = keras.Sequential()
    model.add(layers.Input(shape=(sequence_length, features_count)))

    for layer, kernel in zip(convolution_layers, kernel_sizes):
        model.add(layers.Conv1D(
            filters=layer,
            kernel_size=kernel,
            activation='relu'
        ))

        if spatial_dropout > 0:
            model.add(layers.SpatialDropout1D(rate=spatial_dropout))

        if pooling_type == 'mean':
            model.add(layers.MeanPooling1D(pool_size=pool_size))

        elif pooling_type == 'max':
            model.add(layers.MaxPooling1D(pool_size=pool_size))

        elif pooling_type == 'none':
            pass

    model.add(layers.Flatten())

    for layer in dense_layers:
        model.add(layers.Dense(layer, activation='relu'))

    model.add(layers.Dense(prediction_length))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mse'])
    return model

def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, predicted_label: str, epochs: int, batch_size: int = 32, freq: str = "1h", file_name: str = "cnn_model", early_stopping: bool = True) -> History:
    """
    Trains the provided 1D Convolutional Neural Network (CNN) model using historical time-series data.

    Parameters
    ----------
    model : keras.Sequential
        Compiled Keras CNN model created via `get_model`, ready for training.

    sequence_length : int
        Length of input sequences passed to the model (number of timesteps per sequence).

    prediction_length : int
        Number of timesteps into the future the model predicts.

    predicted_label : str
        Column name containing the target values to be predicted.

    epochs : int
        Number of training epochs.

    batch_size : int, optional (default=32)
        Number of samples per gradient update during training.

    freq : str, optional (default="1h")
        Data frequency used for training (e.g., "1h" for hourly data).

    file_name : str, optional (default="cnn_model")
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
    model_folder = current_folder / 'models'
    model_folder.mkdir(exist_ok=True)

    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS, predicted_label)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split = 0.1 if early_stopping else None,
        callbacks = [early_stop] if early_stopping else [],
        verbose=1)

    model_path = model_folder / f'{file_name}.keras'
    model.save(model_path)

    return history