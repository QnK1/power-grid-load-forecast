import keras
from sklearn.model_selection import train_test_split
from pandas import DataFrame

from .committee import CommitteeSystem
from utils.load_data import load_training_data, DataLoadingParams

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
    X, y = training_data[[col for col in training_data.columns if col != 'load']], training_data['load']
    x_train, x_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.2,
    )
    
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
    # todo: add calling the train_committee method and saving the model to .keras
    pass