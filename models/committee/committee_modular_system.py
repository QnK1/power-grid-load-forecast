import keras
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from pathlib import Path

import tensorflow as tf

from eval.modular_system.modular_system import time_index_from_freq

from .committee import CommitteeSystem
from utils.load_data import load_training_data, DataLoadingParams
from models.modular_system.modular_system import Modular_system, FEATURE_COLUMNS

def train_committee(models_to_wrap: list[keras.Model], training_data: DataFrame) -> keras.Model:
    # get the model
    model = CommitteeSystem(models_to_wrap)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=['mse', 'mape'])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,
        verbose=1,
        restore_best_weights=True 
    )
    
    # get the training and validation data
    X, y = training_data[FEATURE_COLUMNS], training_data['load']
    x_train, x_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.2,
    )
    
    x_train_indices = time_index_from_freq(x_train.index, "1h")
    x_train_indices_tf = tf.convert_to_tensor(x_train_indices, dtype=tf.int32)
    x_val_indices = time_index_from_freq(x_val.index, "1h")
    x_val_indices_tf = tf.convert_to_tensor(x_val_indices, dtype=tf.int32)
    
    x_train = {'features': x_train, 'model_index': x_train_indices_tf}
    x_val = {'features': x_val, 'model_index': x_val_indices_tf}
    
    # train the model
    model.fit(
        x_train, 
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping] 
    )
    
    return model


if __name__ == "__main__":
    MODEL_STRUCTURE = {
        'hidden_layers': [25, 5],
        'n_models': 10,
    }
    
    params = DataLoadingParams()
    params.include_timeindex = True
    data, raw_data = load_training_data(params)
    
    
    submodels = [Modular_system(MODEL_STRUCTURE['hidden_layers'])] * MODEL_STRUCTURE['n_models']
    
    model = train_committee(submodels, data)
    
    model.save(
        Path(__file__).parent.resolve()
        / Path(f'models/committee_modular_system_{MODEL_STRUCTURE["hidden_layers"]}_{MODEL_STRUCTURE["n_models"]}.keras')
    )