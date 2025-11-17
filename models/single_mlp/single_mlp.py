from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from utils.load_data import load_training_data, DataLoadingParams


# ==============================
# CONFIGURATION & UTILITIES
# ==============================

@dataclass
class SingleMLPConfig:
    """
    Configuration for the single-hidden-layer MLP model.

    Attributes
    ----------
    input_dim : int
        Number of input features.
    hidden_units : int
        Number of neurons in the hidden layer.
    learning_rate : float
        Learning rate for the Adam optimizer.
    seed : int
        Random seed used for reproducibility.
    """
    input_dim: int = 16
    hidden_units: int = 25
    learning_rate: float = 1e-3
    seed: int = 42


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and TensorFlow.

    This is used to make experiments as reproducible as possible.

    Parameters
    ----------
    seed : int, optional
        Seed value to be used in all RNGs, by default 42.
    """
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ==============================
# MODEL BUILDING
# ==============================

def build_model(cfg: SingleMLPConfig) -> tf.keras.Model:
    """
    Build a simple feed-forward MLP with a single hidden layer.

    Architecture
    -----------
    Input (cfg.input_dim)
        → Dense(cfg.hidden_units, activation="sigmoid")
        → Dense(1, activation=None)      # linear output

    The model is compiled with mean squared error (MSE) loss
    and the Adam optimizer.

    Parameters
    ----------
    cfg : SingleMLPConfig
        Configuration object specifying input size, number of hidden units
        and learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    inputs = tf.keras.Input(shape=(cfg.input_dim,), name="x")
    x = tf.keras.layers.Dense(
        cfg.hidden_units,
        activation="sigmoid",
        name="hidden"
    )(inputs)
    y = tf.keras.layers.Dense(
        1,
        activation=None,
        name="y"
    )(x)

    model = tf.keras.Model(inputs, y, name="single_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="mse"
    )
    return model


# ==============================
# TRAINING
# ==============================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_units: int = 25,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_split: float = 0.1,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train the single-hidden-layer MLP on the provided training data.

    Early stopping on validation loss is used (if validation_split > 0 and patience > 0).

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (N, D), where N is the number of samples
        and D is the number of features.
    y_train : np.ndarray
        Training targets, shape (N, 1) or (N,).
    hidden_units : int, optional
        Number of units in the hidden layer, by default 25.
    epochs : int, optional
        Maximum number of training epochs, by default 20.
    batch_size : int, optional
        Batch size used during training, by default 256.
    learning_rate : float, optional
        Learning rate for the Adam optimizer, by default 1e-3.
    validation_split : float, optional
        Fraction of the training data used as validation set (0.0–1.0),
        by default 0.1.
    patience : int, optional
        Number of epochs with no improvement in validation loss before
        early stopping, by default 5. If 0, early stopping is disabled.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    (tf.keras.Model, tf.keras.callbacks.History)
        The trained model and the Keras History object.
    """
    set_seeds(seed)
    cfg = SingleMLPConfig(
        input_dim=int(X_train.shape[1]),
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model = build_model(cfg)

    callbacks: list[tf.keras.callbacks.Callback] = []
    if validation_split and patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split if validation_split > 0 else 0.0,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ==============================
# PREDICTION & PERSISTENCE
# ==============================

def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Run inference with a trained model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    X : np.ndarray
        Input features, shape (N, D).

    Returns
    -------
    np.ndarray
        Predicted standardized load values, shape (N, 1).
    """
    return model.predict(X, verbose=0)


def save_model(model: tf.keras.Model, path: str | Path) -> None:
    """
    Save a Keras model to disk in the `.keras` format.

    Parent directories are created if they do not exist.

    Parameters
    ----------
    model : tf.keras.Model
        Model instance to be saved.
    path : str or Path
        Target file path, e.g. "models/single_mlp/single_mlp_25neurons_20epochs.keras".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: str | Path) -> tf.keras.Model:
    """
    Load a previously saved Keras model from disk.

    Parameters
    ----------
    path : str or Path
        Path to a `.keras` model file.

    Returns
    -------
    tf.keras.Model
        Loaded Keras model.
    """
    return tf.keras.models.load_model(Path(path))


# ==============================
# DATA HELPERS
# ==============================

def prepare_Xy(df_ml) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Split a DataFrame into feature matrix X, target vector y and feature names.

    The function assumes that the target column is named 'load' and that all
    remaining columns are feature columns.

    Parameters
    ----------
    df_ml : pandas.DataFrame
        DataFrame returned by the data loader, containing feature columns
        and a 'load' column as the target.

    Returns
    -------
    (np.ndarray, np.ndarray, list[str])
        - X : array of features, shape (N, D), dtype float32
        - y : array of targets, shape (N, 1), dtype float32
        - feature_cols : list of feature column names

    Raises
    ------
    ValueError
        If the 'load' column is missing from the DataFrame.
    """
    if "load" not in df_ml.columns:
        raise ValueError("The DataFrame must contain a 'load' column as the target.")

    feature_cols = [c for c in df_ml.columns if c != "load"]
    X = df_ml[feature_cols].values.astype(np.float32)
    y = df_ml["load"].values.astype(np.float32).reshape(-1, 1)
    return X, y, feature_cols


# ==============================
# SCRIPT ENTRY POINT
# ==============================

if __name__ == "__main__":
    params = DataLoadingParams()
    df_train, df_train_raw = load_training_data(params)

    X_train, y_train, feat_cols = prepare_Xy(df_train)

    # Basic configuration
    hidden_units = SingleMLPConfig().hidden_units
    epochs = 15

    model, hist = train(
        X_train,
        y_train,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
    )

    save_path = f"models/single_mlp/models/single_mlp_{hidden_units}neurons_{epochs}epochs.keras"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_model(model, save_path)
    print(f"Model saved to: {save_path}")
