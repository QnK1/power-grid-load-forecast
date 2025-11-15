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