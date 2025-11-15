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

def get_model(convolution_layers: list[int], kernel_sizes: list[int], lstm_layers: list[int], dense_layers: list[int], sequence_length: int, prediction_length: int, features_count: int, pooling_type: str = 'average', pool_size: int = 2, spatial_dropout: float = 0.1, dropout: float = 0.1, recurrent_dropout: float = 0.1) -> keras.Sequential:
    """
    Builds and compiles a hybrid CNN-LSTM model for multistep time-series forecasting.

    This architecture combines:
    - 1D Convolutional layers for extracting short-term temporal patterns,
    - optional pooling to compress the time dimension,
    - one or more LSTM layers for modeling long-term dependencies after CNN feature extraction,
    - fully-connected layers for final multistep regression.

    The model predicts `prediction_length` future timesteps in a single forward pass
    (direct multi-step forecasting).

    Parameters
    ----------
    convolution_layers : list[int]
        Number of filters for each Conv1D layer.
        Example: [64, 32] creates two convolution layers with 64 and 32 filters.

    kernel_sizes : list[int]
        Kernel size for each convolution layer. Must match the length of `convolution_layers`.
        Example: [3, 2] means kernels of size 3 and 2 for subsequent Conv1D layers.

    lstm_layers : list[int]
        Hidden units for each LSTM layer. At least one LSTM layer is required.
        Only the last LSTM layer returns a single vector; previous ones return sequences.

    dense_layers : list[int]
        Number of units in Dense layers following the LSTM block.
        These layers refine high-level representations before the final output.

    sequence_length : int
        Length of the input sequence (number of historical timesteps).

    prediction_length : int
        Number of future timesteps the model should predict.

    features_count : int
        Number of input features per timestep.

    pooling_type : str, optional (default='average')
        Type of pooling applied after each Conv1D layer:
        - 'average' -> AveragePooling1D
        - 'max'     -> MaxPooling1D
        - 'none'    -> pooling disabled

    pool_size : int, optional (default=2)
        Pooling window size for AveragePooling1D or MaxPooling1D.

    spatial_dropout : float, optional (default=0.1)
        Dropout applied to feature maps after each Conv1D layer.
        Helps reduce overfitting. Set to 0 to disable.

    dropout : float, optional (default=0.1)
        Dropout rate applied inside each LSTM layer.

    recurrent_dropout : float, optional (default=0.1)
        Dropout applied to recurrent connections in LSTM layers.

    Returns
    -------
    keras.Sequential
        A compiled CNN-LSTM model.

    Notes
    -----
    - The model performs direct multi-step forecasting (predicts all future values simultaneously).
    - The optimizer uses gradient clipping for stability during LSTM training.
    """

    if len(convolution_layers) == 0:
        raise ValueError('There should be at least one convolution layer in a CNN-LSTM')

    if len(lstm_layers) == 0:
        raise ValueError('There should be at least one LSTM layer in a CNN-LSTM')

    if len(convolution_layers) != len(kernel_sizes):
        raise ValueError('Each convolution layer should have exactly one kernel size assigned to it.')

    if pooling_type not in ['average', 'max', 'none']:
        raise ValueError('Available types of pooling are: average, max, none.')

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

        if pooling_type == 'average':
            model.add(layers.AveragePooling1D(pool_size=pool_size))

        elif pooling_type == 'max':
            model.add(layers.MaxPooling1D(pool_size=pool_size))

        elif pooling_type == 'none':
            pass

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

def train_model(model: keras.Sequential, sequence_length: int, prediction_length: int, predicted_label: str, epochs: int, batch_size: int = 32, freq: str = "1h", file_name: str = "cnn_lstm_model", early_stopping: bool = True) -> History:
    """
    Trains a hybrid CNN-LSTM model using historical time-series data.

    The training pipeline performs the following steps:
    1. Loads and preprocesses time-series data according to the selected frequency.
    2. Converts the dataset into supervised sequences via `create_sequences`,
       producing:
         - X of shape (num_samples, sequence_length, num_features)
         - y of shape (num_samples, prediction_length)
    3. Trains the provided model using the specified batch size and number of epochs.
    4. Optionally applies EarlyStopping with validation split to prevent overfitting.
    5. Saves the trained model in `.keras` format.
    6. Returns the training history for analysis and visualization.

    Parameters
    ----------
    model : keras.Sequential
        A compiled Keras model created with `get_model`,
        containing convolutional, recurrent or dense layers.

    sequence_length : int
        Number of past timesteps used as input for each training sample.

    prediction_length : int
        Number of future timesteps the model predicts simultaneously.

    predicted_label : str
        Name of the target column to be forecasted (e.g., "load").

    epochs : int
        Maximum number of training epochs.

    batch_size : int, optional (default=32)
        Number of samples processed in each training batch.

    freq : str, optional (default="1h")
        Data sampling frequency used when loading the training dataset.

    file_name : str, optional (default="cnn_lstm_model")
        Base name of the saved model file (`<file_name>.keras`).

    early_stopping : bool, optional (default=True)
        Enables EarlyStopping on validation loss:
        - monitor = "val_loss"
        - min_delta = 0.001
        - patience = 5
        - restore_best_weights = True
        Uses a 10% validation split when enabled.

    Returns
    -------
    History
        Keras History object containing loss and validation loss per epoch.
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

def get_model_name(convolution_layers: list[int], lstm_layers: list[int], dense_layers: list[int], sequence_length: int, prediction_length: int, epochs: int, batch_size: int, freq: str) -> str:
    """
    Generates a standardized model name string based on CNN-LSTM architecture
    and training configuration.

    The naming convention ensures that saved models can be easily identified
    and compared, as the name encodes all relevant hyperparameters.

    Parameters
    ----------
    convolution_layers : list[int]
        List of filter counts for each Conv1D layer, in the order they appear.
        Example: [32, 16] → "CONV_32-16".

    lstm_layers : list[int]
        List of unit counts for each LSTM layer.
        Example: [64, 32] → "LSTM_64-32".

    dense_layers : list[int]
        List of neuron counts for each Dense layer after the recurrent block
        (excluding the final output layer).
        If empty, it is represented as "DENSE_" with no values.

    sequence_length : int
        Number of timesteps used as input for the model (historical window).

    prediction_length : int
        Number of future steps predicted by the model.

    epochs : int
        Number of epochs used during training.

    batch_size : int
        Mini-batch size used during training.

    freq : str
        Data sampling frequency (e.g., "1h" for hourly, "15min" for quarter-hourly).

    Returns
    -------
    str
        A descriptive model identifier encoding the full architecture and
        training configuration.

        Example output:
        "CONV_32-16_LSTM_64-32_DENSE_16-8_168_24_50_32_1h"

        Meaning:
        - Conv1D filters: [32, 16]
        - LSTM units: [64, 32]
        - Dense units: [16, 8]
        - sequence_length = 168
        - prediction_length = 24
        - epochs = 50
        - batch_size = 32
        - frequency = "1h"
    """

    convolution_layers_string = '-'.join(map(str, convolution_layers))
    lstm_layers_string = '-'.join(map(str, lstm_layers))
    dense_layers_string = '-'.join(map(str, dense_layers))

    return f'CONV_{convolution_layers_string}_LSTM_{lstm_layers_string}_DENSE_{dense_layers_string}_{sequence_length}_{prediction_length}_{epochs}_{batch_size}_{freq}'
