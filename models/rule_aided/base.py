import tensorflow as tf
import pandas as pd
import numpy as np

from utils.load_data import load_training_data, DataLoadingParams
from utils.get_easter_days import get_easter_days_for_years


class RuleBasedModel(tf.keras.Model):
    def __init__(self, model_to_wrap, lookup_table, default_value, **kwargs):
        super(RuleBasedModel, self).__init__(**kwargs)
        self.base_model = model_to_wrap
        self.lookup_table = lookup_table
        self.default_value = tf.constant(default_value, dtype=tf.float32)
    
    
    def call(self, inputs: pd.DataFrame):
        """
        Takes the DataFrame input *before* .to_numpy(), as the dates are required for the rules to work.
        """
        # separate ml features from date keys for lookup
        features = inputs.to_numpy()
        rule_keys = inputs.reset_index()
        rule_keys = rule_keys['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        rule_keys = rule_keys.to_numpy()
        
        features = tf.convert_to_tensor(features)
        rule_keys = tf.convert_to_tensor(rule_keys)
        
        base_prediction = self.base_model(features)
        rule_prediction = tf.reshape(self.lookup_table.lookup(rule_keys), [-1, 1])
        
        condition = tf.not_equal(rule_prediction, self.default_value)
        final_output = tf.where(condition, rule_prediction, base_prediction)
        
        return final_output


def get_lookup_table(default_value: float) -> dict[pd.Timestamp, float]:
    # %m-%d format
    SPECIAL_DATES = [
        '01-01', # New Year's
        '01-02', # Day after New Year's
        '12-25', # Christmas
        '12-26', # Christmas
        '05-01', # Labour Day
        '10-03', # German Unity Day
        
        '12-07', # One of the top outliers, close to Nikolaustag
        '01-29', # One of the top outliers, reason unknown
        '01-11', # One of the top outliers, reason unknown
    ]
    YEARS = range(2015, 2051)
    
    rule_averages = {}
    
    df, _ = load_training_data(DataLoadingParams())
    df_date = df.reset_index()
    
    df_date['time_in_year'] = df_date['date'].dt.strftime('%m-%d %H:%M:%S')
    mean_by_time_in_year = pd.DataFrame(df_date.groupby('time_in_year')['load'].mean())
    
    mean_by_time_in_year.reset_index(inplace=True)
    mean_by_time_in_year['simple_time'] = mean_by_time_in_year['time_in_year'].str.slice(0, 5)
    
    for row in mean_by_time_in_year.itertuples():
        if row.simple_time in SPECIAL_DATES:
            for year in YEARS:
                date_str = f"{year}-{row.time_in_year}"
                rule_averages[date_str] = row.load
    
    # calculate mean values for easter days separately
    easter_dates = get_easter_days_for_years(YEARS)
    df_date['date_str'] = df_date['date'].dt.strftime('%Y-%m-%d')
    
    easter_means = {}
    for name, easter_date_list in easter_dates.items():
        to_add = []
        for row in df_date.itertuples():
            if row.date_str in [d.strftime('%Y-%m-%d') for d in easter_date_list]:
                to_add.append(row[0])
        
        easter_means[name] = pd.DataFrame(df_date.iloc[to_add])       
        easter_means[name]['hour_str'] = easter_means[name]['date'].dt.strftime('%H:%M:%S')
        easter_means[name] = easter_means[name].groupby('hour_str')['load'].mean()
    
    
    for name, means in easter_means.items():
        for date in easter_dates[name]:
            for index, value in means.items():
                rule_averages[f"{date.strftime('%Y-%m-%d')} {index}"] = value
    
    keys_tensor = tf.constant(list(rule_averages.keys()), dtype=tf.string)
    values_tensor = tf.constant(list(rule_averages.values()), dtype=tf.float32)
    
    lookup_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
        default_value=default_value,
    )
    
    return lookup_table