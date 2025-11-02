import pandas as pd
import pickle
from pathlib import Path

from models.committee_modular_system.committee_modular_system import CommitteeSystem
from models.modular_system.modular_system_wrapper import ModularSystemWrapper
from utils.load_data import load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs


# FEATURE_COLUMNS = [
#     'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
#     'load_previous_day_timestamp_-2', 'load_previous_day_timestamp_-1',
#     'load_previous_day_timestamp_0', 'load_previous_day_timestamp_1',
#     'load_previous_day_timestamp_2',
#     'prev_3_temperature_timestamps_mean',
#     'prev_day_temperature_5_timestamps_mean',
#     'day_of_week_sin', 'day_of_week_cos',
#     'day_of_year_sin', 'day_of_year_cos'
# ]

def main():
    print("--- Starting Full Training & Evaluation Pipeline ---")
    print("\nStep 1: Initializing models...")
    
    N_MODELS = 10
    
    modular_models = [ModularSystemWrapper(hidden_layers=[[24, 12]], epochs=[20], id=i) for i in range(N_MODELS)]
    committee = CommitteeSystem(models=modular_models)

    df, raw_df = load_training_data(DataLoadingParams())
    
    FEATURE_COLUMNS = [col for col in df.columns if col != 'load']
    
    X, y = df[FEATURE_COLUMNS], df['load']
    
    committee.train(X, y)
    
    file = Path(__file__).parent.resolve() / Path("models/committee_classic.pkl")
    with open(file, "wb") as f:
        pickle.dump(committee, f)

if __name__ == "__main__":
    main()