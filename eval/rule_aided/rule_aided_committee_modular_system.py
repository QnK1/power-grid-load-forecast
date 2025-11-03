import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from utils.load_data import decode_ml_outputs, load_test_data, DataLoadingParams, load_training_data


def eval(model, params):
    params.shuffle = False
    
    df, raw = load_test_data(params)
    _, train_raw = load_training_data(params)
    
    FEATURE_COLS = [col for col in df.columns if col not in ('load', 'hour_of_day_cos', 'hour_of_day_sin')]
    RESULTS_DIR = 'results/committee.json'
    
    X = df[FEATURE_COLS]
    y = raw['load']
    
    X.index = pd.to_datetime(X.index)
    
    size = df.shape[0]
    
    shifted_cols = [col[0] for col in enumerate(df.columns) if 'load_timestamp_' in col[1]]
    
    print(shifted_cols)
    
    mapes = {}
    for i in range(size - 24):
        pred = []
        buffer = X.iloc[i:i+24].copy()
        
        start_hour = X.index[i].hour
        
        for j in range(0, 24):
            curr_pred = model.predict(buffer.iloc[j:j+1])
            
            pred.append(np.squeeze(curr_pred, axis=0))
            
            for isc, sc in enumerate(shifted_cols):
                if j + isc + 1 < 24:
                    buffer.iloc[j + isc + 1, sc] = curr_pred
        
        mape = mean_absolute_percentage_error(y.iloc[i:i+24], decode_ml_outputs(pred, train_raw))
        mapes.setdefault(start_hour, []).append(mape)
        
        print(f"{i} / {size}, {mape}")
    
    final_mapes = {}
    for hour, mape_list in mapes.items():
        final_mapes[hour] = np.mean(mape_list)
    
    with open(Path(__file__).parent / Path(RESULTS_DIR), 'w') as f:
        json.dump(final_mapes, f, indent=4)



if __name__ == "__main__":
    model_file = Path(__file__).parent.parent.parent.resolve() / Path('models/rule_aided/models/committee_classic.pkl')
    
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
        
        params = DataLoadingParams()
        
        eval(model, params)