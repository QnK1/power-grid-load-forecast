import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import load_model

from utils.load_data import decode_ml_outputs, load_test_data, DataLoadingParams, load_training_data
from models.rule_aided.base import RuleBasedModel, preprocess_inputs
from eval.result_plotter import plot_predictions


def eval(model, params):
    params.shuffle = False
    
    df, raw = load_test_data(params)
    _, train_raw = load_training_data(params)
    
    FEATURE_COLS = [col for col in df.columns if col not in ('load', 'hour_of_day_cos', 'hour_of_day_sin')]
    RESULTS_DIR = 'committee_10_dense_24-12_v2'
    
    X = df[FEATURE_COLS]
    y = raw['load']
    
    X.index = pd.to_datetime(X.index)
    
    size = df.shape[0]
    
    shifted_cols = [col[0] for col in enumerate(df.columns) if 'load_timestamp_' in col[1]]
    
    mapes = {}
    preds = {}
    for i in range(size - 24):
        pred = []
        buffer = X.iloc[i:i+24].copy()
        
        start_hour = X.index[i].hour
        
        for j in range(0, 24):
            print(buffer.iloc[j:j+1])
            print(type(buffer.iloc[j:j+1]))
            
            curr_pred = model.predict(preprocess_inputs(buffer.iloc[j:j+1]))
            
            pred.append(np.squeeze(curr_pred, axis=0))
            
            for isc, sc in enumerate(shifted_cols):
                if j + isc + 1 < 24:
                    buffer.iloc[j + isc + 1, sc] = curr_pred
        
        decoded_pred = decode_ml_outputs(pred, train_raw)
        mape = mean_absolute_percentage_error(y.iloc[i:i+24], decoded_pred)
        preds.setdefault(start_hour, []).extend(decoded_pred)
        mapes.setdefault(start_hour, []).append(mape)
        
        print(f"{i} / {size}, {mape}")
    
    final_mapes = {}
    for hour, mape_list in mapes.items():
        final_mapes[hour] = np.mean(mape_list)
    
    with open(Path(__file__).parent / Path(f"results/{RESULTS_DIR}.json"), 'w') as f:
        json.dump(final_mapes, f, indent=4)
    
    plot_predictions(y, preds[0], [i for i in range(y.shape[0])], RESULTS_DIR, True)



if __name__ == "__main__":
    model_file = Path(__file__).parent.parent.parent.resolve() / Path('models/rule_aided/models/rule_aided_committee.keras')
    model = load_model(model_file)
    
    params = DataLoadingParams()
        
    eval(model, params)