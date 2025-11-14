from keras.models import load_model
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd

from eval.modular_system.modular_system import time_index_from_freq
from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.modular_system.modular_system import FEATURE_COLUMNS, get_model
from models.committee.committee import CommitteeSystem
from eval.helpers import eval_autoregressive
from analysis.model_analysis import ModelPlotCreator


def prepare_inputs(x: pd.DataFrame):
    x_indices = time_index_from_freq(x.index, "1h")
    x_indices_tf = tf.convert_to_tensor(x_indices, dtype=tf.int32)

    return {'features': x, 'model_index': x_indices_tf}


if __name__ == "__main__":
    PATH = Path(__file__).parent.parent.parent.resolve() / Path('models/committee/models') 
    MODEL_PATH = PATH / Path('committee_modular_system_[25, 5]_10.keras')
    
    params = DataLoadingParams()
    params.include_timeindex = True
    # params.include_is_dst = True
    params.shuffle = False
    
    model = load_model(MODEL_PATH)
    
    pred, mapes = eval_autoregressive(model, params, FEATURE_COLUMNS, 24, prepare_inputs)
    mape_total = sum(mapes.values()) / len(mapes.values())
    
    MODEL_NAME = f'{MODEL_PATH}'.replace(f'{PATH}\\', '').replace(".keras", "")
    with open(f"{Path(__file__).parent.resolve() / Path('results')}/{MODEL_NAME}.txt", 'w') as f:
        f.write('starting_hour, mape\n')
        for hour, mape in sorted(mapes.items()):
            f.write(f'{hour}, {mape}\n')
        f.write(f'total, {mape_total}\n')
    
    
    model_plot_creator = ModelPlotCreator()
    
    BEST_START = 7
    
    _, raw_test = load_test_data(params)
    non_zero_mask = pred[BEST_START] != 0.0
    
    model_plot_creator.plot_predictions(raw_test['load'].iloc[non_zero_mask].iloc[0:24].to_numpy(), pred[BEST_START][non_zero_mask][0:24], f'{MODEL_NAME}-Ex1', 'committee_modular_system', "1h")
    model_plot_creator.plot_predictions(raw_test['load'].iloc[non_zero_mask].iloc[1000:1024].to_numpy(), pred[BEST_START][non_zero_mask][1000:1024], f'{MODEL_NAME}-Ex2', 'committee_modular_system', "1h")
    model_plot_creator.plot_predictions(raw_test['load'].iloc[non_zero_mask].iloc[500:524].to_numpy(), pred[BEST_START][non_zero_mask][500:524], f'{MODEL_NAME}-Ex3', 'committee_modular_system', "1h")
    
    horizon_indices = {i: range(i, len(raw_test['load'].iloc[non_zero_mask]), 24) for i in range(24)}
    
    horizon_mapes = [
        100 * mean_absolute_percentage_error(raw_test['load'].iloc[non_zero_mask].iloc[horizon_indices[i]], pred[BEST_START][non_zero_mask][horizon_indices[i]])
        for i in range(24)
    ]
    
    model_plot_creator.plot_mape_over_horizon(MODEL_NAME, horizon_mapes, "committee_modular_system")