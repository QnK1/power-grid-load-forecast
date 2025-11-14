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