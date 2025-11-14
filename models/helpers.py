import numpy as np
import pandas as pd
import sklearn

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