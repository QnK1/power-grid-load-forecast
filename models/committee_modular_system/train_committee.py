import pandas as pd
from models.committee_modular_system.committee_modular_system import CommitteeSystem
from models.modular_system.modular_system_wrapper import ModularSystemWrapper
from utils.load_data import load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs

from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

FEATURE_COLUMNS = [
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
    'load_previous_day_timestamp_-2', 'load_previous_day_timestamp_-1',
    'load_previous_day_timestamp_0', 'load_previous_day_timestamp_1',
    'load_previous_day_timestamp_2',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def main():
    print("--- Starting Full Training & Evaluation Pipeline ---")
    print("\nStep 1: Initializing models...")
    modular_model = ModularSystemWrapper(hidden_layers=[[24, 12]], epochs=[20])
    committee = CommitteeSystem(models=[modular_model])

    df, raw_df = load_training_data(DataLoadingParams())
    
    X, y = df[FEATURE_COLUMNS], df['load']
    
    committee.train(X, y)
    
    params = DataLoadingParams()
    params.shuffle = False
    test_df, test_df_raw = load_test_data(params)
    
    pred = committee.predict(test_df[FEATURE_COLUMNS])
    pred = decode_ml_outputs(pred, raw_df)
    
    mape = mean_absolute_percentage_error(test_df_raw['load'].to_numpy(), pred)
    
    print(test_df_raw['load'])
    print(pred)
    print(mape)

    
    test_df_raw.to_csv("dff.csv")

if __name__ == "__main__":
    main()