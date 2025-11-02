from utils.load_data import load_training_data, DataLoadingParams

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

# ==============================
# KONFIGURACJA I POMOCNICZE
# ==============================

@dataclass
class SingleMLPConfig:
    """Konfiguracja jednowarstwowego MLP (sigmoid)."""
    input_dim: int = 16
    hidden_units: int = 12
    learning_rate: float = 1e-3
    seed: int = 42


def set_seeds(seed: int = 42) -> None:
    """Ustalanie ziarna losowego (dla powtarzalności wyników)."""
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ==============================
# BUDOWA MODELU
# ==============================

def build_model(cfg: SingleMLPConfig) -> tf.keras.Model:
    """
    Architektura:
      Input → Dense(sigmoid, hidden_units) → Dense(linear, 1)
    """
    inputs = tf.keras.Input(shape=(cfg.input_dim,), name="x")
    x = tf.keras.layers.Dense(cfg.hidden_units, activation="sigmoid", name="hidden")(inputs)
    y = tf.keras.layers.Dense(1, activation=None, name="y")(x)
    model = tf.keras.Model(inputs, y, name="single_mlp")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    return model


# ==============================
# TRENING
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
    Trenuje Single-MLP.
    """
    set_seeds(seed)
    cfg = SingleMLPConfig(input_dim=int(X_train.shape[1]), hidden_units=hidden_units, learning_rate=learning_rate)
    model = build_model(cfg)

    callbacks = []
    if validation_split and patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
            )
        )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split if validation_split > 0 else 0.0,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ==============================
# PREDYKCJA I ZAPIS
# ==============================

def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Zwraca przewidywany standaryzowany 'load' (kształt [N,1])."""
    return model.predict(X, verbose=0)


def save_model(model: tf.keras.Model, path: str | Path) -> None:
    """Zapisz model w formacie .keras."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: str | Path) -> tf.keras.Model:
    """Wczytaj wcześniej zapisany model."""
    return tf.keras.models.load_model(Path(path))


# ==============================
# POMOCNICZE DO DANYCH
# ==============================

def prepare_Xy(df_ml) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Przyjmuje DataFrame zwrócony przez Twój loader (kolumny z cechami + 'load').
    Zwraca: X, y, lista_nazw_cech.
    """
    if "load" not in df_ml.columns:
        raise ValueError("W DataFrame musi być kolumna 'load' (target).")

    feature_cols = [c for c in df_ml.columns if c != "load"]
    X = df_ml[feature_cols].values.astype(np.float32)
    y = df_ml["load"].values.astype(np.float32).reshape(-1, 1)
    return X, y, feature_cols


if __name__ == "__main__":
    from utils.load_data import load_training_data, DataLoadingParams

    print("Ładowanie danych treningowych...")
    params = DataLoadingParams()
    df_train, df_train_raw = load_training_data(params)

    print("Przygotowywanie macierzy X/y...")
    X_train, y_train, feat_cols = prepare_Xy(df_train)
    print(f"Dane gotowe: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")

    # konfiguracja
    hidden_units = SingleMLPConfig().hidden_units
    epochs = 20

    print(f"Trenowanie modelu ({hidden_units} neuronów, {epochs} epok)...")
    model, hist = train(
        X_train, y_train,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
    )

    # automatyczna ścieżka zapisu
    save_path = f"models/single_mlp/trained/single_mlp_{hidden_units}neurons_{epochs}epochs.keras"
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_model(model, save_path)
    print(f"Model zapisany: {save_path}")

