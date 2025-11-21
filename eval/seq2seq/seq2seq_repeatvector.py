from pathlib import Path
from keras.models import load_model
import numpy as np

from analysis.model_analysis import ModelPlotCreator
from models.helpers import create_sequences
from utils.load_data import DataLoadingParams, decode_ml_outputs, load_test_data, load_training_data
from models.seq2seq.seq2seq import Seq2SeqRepeatVectorModel


def evaluate_model(sequence_length: int, prediction_length: int, model_name: str, save_plots: bool):
    FEATURE_COLUMNS = [
        'load',
        'prev_3_temperature_timestamps_mean',
        'prev_day_temperature_5_timestamps_mean',
        'hour_of_day_sin', 'hour_of_day_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'timestamp_day'
    ]
    
    project_folder = Path(__file__).parent.parent.parent
    model_path = project_folder / 'models' / 'seq2seq' / 'models' / f'{model_name}.keras'
    # model = Seq2SeqRepeatVectorModel(24, [32, 32], [32, 32], True, 0.2)
    # model.build(input_shape=(None, 168, 10))
    # model.load_weights(model_path)
    model = load_model(model_path)

    params = DataLoadingParams()
    params.freq = "1h"
    params.shuffle = False
    params.interpolate_empty_values = True
    params.include_timeindex = True
    
    test_data, _ = load_test_data(params)
    _, raw_data = load_training_data(params)
    X_test, y_test = create_sequences(test_data, sequence_length, prediction_length, FEATURE_COLUMNS, 'load', shuffle=False)
    
    predictions = model.predict(X_test)
    y_real = decode_ml_outputs(y_test, raw_data)
    y_pred = decode_ml_outputs(np.squeeze(predictions, -1), raw_data)
    
    mapes = []
    for start in range(y_real.shape[0]):
        real = y_real[start, :]
        pred = y_pred[start, :]
        
        mask = real != 0
        
        if np.any(mask):
            mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100
        else:
            mape = np.nan

        mapes.append(mape)
    
    mape_per_starting_hour = []
    for i in range(24):
        indices = list(range(i, len(mapes), 24))
        mape_per_starting_hour.append(np.mean(np.array(mapes)[indices]))
    
    mape_per_step = []
    
    for step in range(y_real.shape[1]):
        real_step = y_real[:, step]
        pred_step = y_pred[:, step]

        mask = real_step != 0

        if np.any(mask):
            mape = np.mean(np.abs((real_step[mask] - pred_step[mask]) / real_step[mask])) * 100
        else:
            mape = np.nan

        mape_per_step.append(mape)

    mape_per_step = np.array(mape_per_step, dtype=float)
    mape_total = np.nanmean(mape_per_step)

    with open(f'{Path(__file__).parent}/results/eval_results_{model_name}.txt', 'w') as f:
        f.write(f'starting_hour, mape_1h\n')
        for strat, mape_start in enumerate(mape_per_starting_hour):
            f.write(f'{strat}, {mape_start}\n')
        f.write(f'total, {mape_total}\n')
    
    plot_creator = ModelPlotCreator()
    
    plot_creator.plot_mape_over_horizon(model_name, mape_per_step, "seq2seq", save_plot=save_plots, show_plot=True)
    
    plot_creator = ModelPlotCreator()
    examples = np.random.randint(0, 1000, 3)
    for i, example in enumerate(examples):
        plot_creator.plot_predictions(y_real[example], y_pred[example],  f'{model_name}-Ex{i+1}', "seq2seq", freq='1h', save_plot=save_plots, show_plot=True)
        

if __name__ == "__main__":
    evaluate_model(168, 24, "SEQ2SEQ_RV_168_24_[32, 32]_[32, 32]_bidirectional", True)