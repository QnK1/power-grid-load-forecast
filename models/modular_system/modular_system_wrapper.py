import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from .modular_system import create_and_train_models

class ModularSystemWrapper:
    def __init__(self, hidden_layers=[[24, 12]], epochs=[20], freq="1h"):
        self.name = "Modular System"
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.freq = freq
        self.models_dir = Path(__file__).parent / "models"
        print(f"Initialized {self.name} Wrapper.")

    def train(self, X_train, y_train):
        print(f"   Handing off training to original create_and_train_models function...")
        create_and_train_models(self.hidden_layers, self.epochs, self.freq)

    def predict(self, X):
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