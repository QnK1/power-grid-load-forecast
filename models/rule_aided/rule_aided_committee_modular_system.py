from utils.load_data import load_training_data, DataLoadingParams
from .base import get_lookup_table, get_special_days_dict

def train_base_model():
    params = DataLoadingParams()
    
    df, raw = load_training_data(params)
    
    special_days_dict = get_special_days_dict()