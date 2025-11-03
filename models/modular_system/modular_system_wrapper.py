import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from .modular_system import train_models

class ModularSystemWrapper:
    def __init__(self, hidden_layers=[[24, 12]], epochs=[20], freq="1h", id=0, drop_special_days=False):
        self.name = "Modular System"
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.freq = freq
        self.models_dir = Path(__file__).parent / "models"
        self.id = id
        self.hourly_models = {}
        self.drop_special_days = drop_special_days
        

    def train(self, X_train, y_train):
        print(f"   Handing off training to original train_models function...")
        all_loaded = True
        
        for hour in range(24):
            hidden_str = "-".join(map(str, self.hidden_layers[0]))
            model_filename = f"model_{hour}_{hidden_str}_{self.epochs[0]}_{self.freq}_{self.id}.keras"
            model_path = self.models_dir / model_filename
            
            if model_path.exists():
                # Load the model and store it in the dictionary
                self.hourly_models[hour] = tf.keras.models.load_model(model_path, compile=False)
            else:
                # Store None if a model for a specific hour doesn't exist
                self.hourly_models[hour] = None
                all_loaded = False
                break
        
        
        if not all_loaded:
            train_models(self.hidden_layers, self.epochs, self.freq, self.id, self.drop_special_days)
            
            for hour in range(24):
                hidden_str = "-".join(map(str, self.hidden_layers[0]))
                model_filename = f"model_{hour}_{hidden_str}_{self.epochs[0]}_{self.freq}_{self.id}.keras"
                model_path = self.models_dir / model_filename
                
                if model_path.exists():
                    # Load the model and store it in the dictionary
                    self.hourly_models[hour] = tf.keras.models.load_model(model_path, compile=False)


    def predict(self, X):
        """
        Public-facing predict method.
        Handles data conversion from pandas to TensorFlow tensors.
        """
        x_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)
        hours_tensor = tf.convert_to_tensor(X.index.hour, dtype=tf.int32)

        predictions = self._predict_tf(x_tensor, hours_tensor)

        return predictions.numpy().reshape(-1, 1)


    @tf.function
    def _predict_tf(self, x_tensor, hours_tensor):
        """
        High-performance prediction logic compiled into a TensorFlow graph.
        """
        num_inputs = tf.shape(x_tensor)[0]
        original_indices = tf.range(num_inputs)
        
        predictions_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        indices_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        
        array_idx = 0

        for hour in range(24):
            model = self.hourly_models.get(hour)
            if model is not None:
                hour_mask = tf.equal(hours_tensor, hour)
                hourly_data = tf.boolean_mask(x_tensor, hour_mask)
                hourly_indices = tf.boolean_mask(original_indices, hour_mask)

                if tf.shape(hourly_data)[0] > 0:
                    predictions = model(hourly_data, training=False)

                    reshaped_predictions = tf.reshape(predictions, [-1])
                    
                    predictions_array = predictions_array.write(array_idx, reshaped_predictions)
                    indices_array = indices_array.write(array_idx, hourly_indices)
                    array_idx += 1
        
        all_predictions = predictions_array.concat()
        all_indices = indices_array.concat()
        
        final_predictions = tf.scatter_nd(
            indices=tf.expand_dims(all_indices, axis=1),
            updates=all_predictions,
            shape=tf.cast([num_inputs], dtype=tf.int32)
        )

        return final_predictions