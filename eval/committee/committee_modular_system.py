from keras.models import load_model
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

from eval.modular_system.modular_system import time_index_from_freq
from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs
from models.modular_system.modular_system import FEATURE_COLUMNS, get_model
from models.committee.committee import CommitteeSystem

def eval(model, params):
    test_data, raw_test = load_test_data(params)
    _, raw_trainig = load_training_data(params)
    
    X, y = test_data[FEATURE_COLUMNS], raw_test['load']
    
    lag_indices = [i[0] for i in enumerate(X.columns) if 'load_timestamp' in i[1]]
    
    mapes = {}
    size = X.shape[0]
    
    for i in range(len(X) - 24):
        starting_hour = X.index[i].hour
        
        x_copy = X.iloc[i:i+24].copy()
        
        pred = []
        y_real = y.iloc[i:i+24]
        
        for j in range(24):
            x_indices = time_index_from_freq(x_copy.iloc[[j]].index, "1h")
            x_indices_tf = tf.convert_to_tensor(x_indices, dtype=tf.int32)

            input = {'features': x_copy.iloc[[j]], 'model_index': x_indices_tf}

            output = model.predict(input, verbose=0)
            pred.append(decode_ml_outputs(output, raw_trainig))
            
            for lag_number, lag_value in enumerate(lag_indices):
                if j + lag_number + 1 + 1 < 24:
                    x_copy.iloc[j + lag_number + 1, lag_value] = output
        
        mape = mean_absolute_percentage_error(y_real, np.squeeze(pred, axis=1))
        mapes.setdefault(starting_hour, []).append(mape)
        
        print(f"{i} / {size}, {mape}")
    
    final_mapes = {}
    for hour, mape_list in mapes.items():
        final_mapes[hour] = np.mean(mape_list)
        
    return final_mapes
    


if __name__ == "__main__":
    PATH = Path(__file__).parent.parent.parent.resolve() / Path('models/committee/models') 
    MODEL_PATH = PATH / Path('committee_modular_system_[5, 9, 12, 18, 22, 25]_10.keras')
    
    params = DataLoadingParams()
    params.include_timeindex = True
    params.shuffle = False
    
    model = load_model(MODEL_PATH)
    
    mapes = eval(model, params)
    mape_total = sum(mapes.values()) / len(mapes.values())
    
    with open(f"{Path(__file__).parent.resolve() / Path('results')}/{f'{MODEL_PATH}'.replace(f'{PATH}', '')}", 'w') as f:
        f.write('starting_hour, mape\n')
        for hour, mape in sorted(mapes.items()):
            f.write(f'{hour}, {mape}\n')
        f.write(f'total, {mape_total}\n')