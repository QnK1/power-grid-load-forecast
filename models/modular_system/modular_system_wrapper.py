import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from .modular_system import create_and_train_models

class ModularSystemWrapper:
    """
    Wraps the original script to make it compatible with the CommitteeSystem.
    """
    def __init__(self, hidden_layers=[[24, 12]], epochs=[20], freq="1h"):
        self.name = "Modular System"
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.freq = freq
        self.models_dir = Path(__file__).parent / "models"
        print(f"Initialized {self.name} Wrapper.")

    def train(self, X_train, y_train):
        """Calls the original script to create and save the 24 hourly models."""
        print(f"   Handing off training to original create_and_train_models function...")
        create_and_train_models(
            hidden_layers=self.hidden_layers,
            epochs=self.epochs,
            freq=self.freq
        )

    def predict(self, X):
        """Makes predictions by loading the correct saved hourly model."""
        all_predictions = pd.Series(index=X.index, dtype=float)
        for hour, group in X.groupby(X.index.hour):
            if group.empty: continue
            hidden_str = "-".join(map(str, self.hidden_layers[0]))
            model_filename = f"model_{hour}_{hidden_str}_{self.epochs[0]}_{self.freq}.h5"
            model_path = self.models_dir / model_filename
            if not model_path.exists():
                all_predictions.loc[group.index] = np.nan
                continue
            hourly_model = tf.keras.models.load_model(model_path, compile=False)
            predictions = hourly_model.predict(group, verbose=0)
            all_predictions.loc[group.index] = predictions.flatten()
        return all_predictions.fillna(0).values.reshape(-1, 1)