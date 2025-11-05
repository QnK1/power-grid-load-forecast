import tensorflow as tf
import pandas as pd
import numpy as np
import keras

from utils.load_data import load_training_data, DataLoadingParams
from utils.get_easter_days import get_easter_days_for_years

@keras.saving.register_keras_serializable()
class RuleBasedModel(tf.keras.Model):
    def __init__(self, model_to_wrap, lookup_table, values_tensor, **kwargs):
        super(RuleBasedModel, self).__init__(**kwargs)
        self.base_model = model_to_wrap
        self.lookup_table = lookup_table
        self.values_tensor = values_tensor
        self.default_value_tensor = tf.constant([-1000, -1000], dtype=tf.float64)
        
    
    def call(self, inputs: tf.Tensor):
        features = inputs['features']
        prev_load_real = inputs['prev_load_real']
        rule_keys = inputs['rule_keys']
        
        base_prediction = self.base_model(features)
        
        looked_up_indices = self.lookup_table.lookup(rule_keys)
        valid_indices = tf.maximum(looked_up_indices, 0)
        gathered_values = tf.gather(self.values_tensor, valid_indices)
        
        mask = tf.not_equal(looked_up_indices, tf.constant(-1, dtype=tf.int64))
        
        lookup_values = tf.where(
            tf.expand_dims(mask, axis=-1),
            gathered_values,
            self.default_value_tensor
        )
        
        to_subtract = tf.Variable(lookup_values[:, 1]).assign_sub(prev_load_real).value()
        rule_prediction = tf.reshape(tf.Variable(lookup_values[:, 0]).assign_sub(to_subtract).value(), [-1, 1])
        rule_prediction = tf.cast(rule_prediction, dtype=tf.float32)
        
        condition = tf.expand_dims(mask, axis=-1)
        final_output = tf.where(condition, rule_prediction, base_prediction)
        
        return final_output
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "base_model": keras.saving.serialize_keras_object(self.base_model)
        })
        return config


    @classmethod
    def from_config(cls, config):
        config["base_model"] = keras.saving.deserialize_keras_object(config["base_model"])
        
        return cls(**config)


def preprocess_inputs(inputs: pd.DataFrame):
    features_tensor = tf.convert_to_tensor(inputs.to_numpy())
    keys_tensor = tf.convert_to_tensor(inputs.reset_index()['date'].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy())
    prev_load_tensor = tf.convert_to_tensor(inputs['load_timestamp_-1'].to_numpy())
    
    return {
        'features': features_tensor,
        'prev_load_real': prev_load_tensor,
        'rule_keys': keys_tensor,
    }


def get_lookup_table() -> tuple[tf.lookup.StaticHashTable, tf.Tensor]:
    rule_averages = get_special_days_dict()
    
    np_array = np.array(list(rule_averages.values()))
    values_tensor = tf.convert_to_tensor(np_array)
    
    keys_tensor = tf.constant(list(rule_averages.keys()), dtype=tf.string)
    
    indices = tf.range(start=0, limit=len(rule_averages.keys()), dtype=tf.int64)
    default_index = tf.constant(-1, dtype=tf.int64)
    
    table_init = tf.lookup.KeyValueTensorInitializer(keys_tensor, indices)
    table = tf.lookup.StaticHashTable(table_init, default_value=default_index)
    
    return table, values_tensor


def get_special_days_dict():
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
    
    # to use for potential missing values
    global_mean = mean_by_time_in_year['load'].mean()
    
    prev_row = None
    for row in mean_by_time_in_year.itertuples():
        if row.simple_time in SPECIAL_DATES:
            for year in YEARS:
                date_str = f"{year}-{row.time_in_year}"
                
                if prev_row is None:
                    rule_averages[date_str] = np.float64(row.load), global_mean
                else:
                    rule_averages[date_str] = np.float64(row.load), prev_row.load
        
        prev_row = row
    
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
                rule_averages[f"{date.strftime('%Y-%m-%d')} {index}"] = np.float64(value), mean_by_time_in_year.set_index('time_in_year').loc[f"{(date - pd.Timedelta(days=1)).strftime('%m-%d')} {index}"]['load']
                
    return rule_averages