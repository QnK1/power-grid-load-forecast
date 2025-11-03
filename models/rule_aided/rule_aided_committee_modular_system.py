import pandas as pd
from pathlib import Path
import pickle

from models.committee_modular_system.committee_modular_system import CommitteeSystem
from models.modular_system.modular_system_wrapper import ModularSystemWrapper
from utils.load_data import load_training_data, DataLoadingParams
from .base import RuleBasedModel, get_lookup_table, get_special_days_dict
from models.modular_system.modular_system import get_model

COMMITTEE_PATH = Path(__file__).parent.resolve() / Path("models/committee_classic.pkl")

def train_base_model(hidden_layers, epochs, n_committee):
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
    
    params = DataLoadingParams()
    
    df, raw = load_training_data(params)
    
    print(len(df), len(raw))
    
    special_days_dict = get_special_days_dict()
    special_dates = [pd.to_datetime(k) for k in special_days_dict.keys()]
    
    df, raw = df.reset_index(), raw
    df, raw = df[~df['date'].isin(special_dates)], raw[~raw['date'].isin(special_dates)]
    df, raw = df.set_index('date'), raw.set_index('date')
    
    
    modular_models = [ModularSystemWrapper(hidden_layers=[[24, 12]], epochs=[20], id=i, drop_special_days=True) for i in range(n_committee)]
    committee = CommitteeSystem(models=modular_models)
    
    X, y = df[FEATURE_COLUMNS], df['load']
    
    committee.train(X, y)
    
    file = COMMITTEE_PATH
    with open(file, "wb") as f:
        pickle.dump(committee, f)


def get_rule_aided_model(base_model):
    DEFAULT_VALUE = -1000.0
    MODEL_PATH = Path(__file__).parent.resolve() / Path('models')
    
    lookup_table = get_lookup_table(DEFAULT_VALUE)
    
    model = RuleBasedModel(base_model, lookup_table, DEFAULT_VALUE)
    
    model.save(MODEL_PATH / Path('rule_aided_committee.keras'))



if __name__ == "__main__":
    train_base_model(hidden_layers=[[24, 12]], epochs=25, n_committee=10)
    
    with open(COMMITTEE_PATH, 'rb') as file:
        base_model = pickle.load(file)
        
        get_rule_aided_model(base_model)