import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras import Model
from sklearn.metrics import mean_absolute_percentage_error
from typing import Callable

from utils.load_data import load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs


def eval_autoregressive(model: Model, params: DataLoadingParams, feature_columns: list[str], horizon: int = 24, prepare_inputs: Callable = None) -> tuple[dict[int, np.ndarray], dict[int, np.float64]]:
    """An optimized multi-step evalutaion function for models that only predict one step into the future. 
    Args:
        model (Model): The (trained) model to evaluate.
        params (DataLoadingParams): The DataLoadingParams that were used to load the training data for the model.
        feature_columns (list[str]): The list of feature columns the model uses.
        horizon (int, optional): How many steps into the future should the model be evaluated on
                                    (e.g. 24 for a 24h evaluation for a model with a 1h freq).
        prepare_inputs (Callable, optional): A function used to convert simple data rows into data that the model accepts.
                                                Useful for advanced models that use dictionaries as inputs. The function should
                                                be able to work on batches, not only single samples. 
                                                I None, standard data rows will be passed to the model.

    Returns:
        tuple[dict[int, np.ndarray], dict[int, np.float64]]: A dictionary of predictions for each starting hour,
                                                                as well as a dictionary of mean MAPEs for each starting hour.
    """
    test_data, raw_test = load_test_data(params)
    _, raw_trainig = load_training_data(params)
    
    X, y = test_data[feature_columns], raw_test['load']
    
    lag_indices = [i[0] for i in enumerate(X.columns) if 'load_timestamp' in i[1]]
    size = X.shape[0]
    max_startingpoint = size - horizon
    
    base_starting_points = np.array(range(0, max_startingpoint + 1, horizon))
    
    mapes = {}
    pred = {i: np.zeros(size) for i in range(horizon)}
    
    for i in range(horizon):
        x_copy = X.copy()
        
        starting_points = base_starting_points + i
        
        current_start = x_copy.index[starting_points[0]].hour
        
        for j in range(horizon):
            current_indices = starting_points + j
            valid_j_mask = current_indices < size
            if not np.any(valid_j_mask):
                break
            current_indices_valid = current_indices[valid_j_mask]
            
            if prepare_inputs is not None:
                inputs = prepare_inputs(x_copy.iloc[current_indices_valid])
            else:
                inputs = x_copy.iloc[current_indices_valid]
                
            outputs = model.predict(inputs, verbose=0)
            outputs_squeezed = np.squeeze(outputs)
            if outputs_squeezed.ndim == 0:
                outputs_squeezed = np.array([outputs])
            
            curr_pred = decode_ml_outputs(outputs, raw_trainig)
            
            for lag_index_number, lag_index_value in enumerate(lag_indices):
                target_indices = current_indices_valid + lag_index_number + 1
                valid_target_mask = target_indices < size
                indices_to_update = target_indices[valid_target_mask]
                values_to_write = outputs_squeezed[valid_target_mask]
                
                if len(indices_to_update) > 0:
                    x_copy.iloc[indices_to_update, lag_index_value] = values_to_write
            
            curr_mape = mean_absolute_percentage_error(y.iloc[current_indices_valid], curr_pred)
            mapes.setdefault(current_start, []).append(curr_mape)
            pred[i][current_indices_valid] = np.squeeze(curr_pred)
            
            print(f"{i, j} / {horizon, horizon}, {curr_mape}")
    
    
    mean_mapes = {}
    for hour, mape_list in mapes.items():
        mean_mapes[hour] = np.mean(mape_list)
    
    return pred, mean_mapes