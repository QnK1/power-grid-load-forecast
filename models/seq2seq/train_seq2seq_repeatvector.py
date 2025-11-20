import tensorflow as tf
import keras
from pathlib import Path

from models.helpers import create_sequences
from .seq2seq import Seq2SeqRepeatVectorModel
from utils.load_data import load_training_data, DataLoadingParams

def train_seq2seq_repeatvector(sequence_length: int, prediction_length: int, encoder_layers: list[int], decoder_layers: list[int], bidirectional: bool, dropout: float):
    FEATURE_COLUMNS = [
        'load',
        'prev_3_temperature_timestamps_mean',
        'prev_day_temperature_5_timestamps_mean',
        'hour_of_day_sin', 'hour_of_day_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'timestamp_day'
    ]
    
    model = Seq2SeqRepeatVectorModel(
        prediction_length=prediction_length,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        bidirectional=bidirectional,
        dropout=dropout
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=['mse'])
    
    params = DataLoadingParams()
    params.freq = "1h"
    params.interpolate_empty_values = True
    params.shuffle = False
    params.include_timeindex = True
    
    data, _ = load_training_data(params)
    
    X_train, y_train = create_sequences(data, sequence_length, prediction_length, FEATURE_COLUMNS, "load")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,
        verbose=1,
        restore_best_weights=True 
    )
    
    model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model


if __name__ == "__main__":
    MODEL_PARAMS = {
        'sequence_length': 168,
        'prediction_length': 24,
        'encoder_layers': [32, 32],
        'decoder_layers': [32, 32],
        'bidirectional': True,
        'dropout': 0.2,
    }
    
    current_dir = Path(__file__).parent
    model_dir = current_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    model_name = (
        f"SEQ2SEQ_RV_{MODEL_PARAMS['sequence_length']}_"
        f"{MODEL_PARAMS['prediction_length']}_"
        f"{MODEL_PARAMS['encoder_layers']}_"
        f"{MODEL_PARAMS['decoder_layers']}_"
        f"{'bidirectional' if MODEL_PARAMS['bidirectional'] else 'unidirectional'}"
    )
    
    model = train_seq2seq_repeatvector(**MODEL_PARAMS)
    
    model.save(model_dir / f"{model_name}.keras")