import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
from tensorflow import Tensor

LOAD_DATA_RANDOM_STATE = int(np.random.default_rng().random() * 100)

class DataLoadingParams:
    """Customizable parameters for the load_data function.

    To use defaults from the original paper, don't modify anything,
    just use:
    df, raw_df = load_data(DataLoadingParams())

    To modify parameters, use, for example:
    params = DataLoadingParams()
    params.freq = "2h"
    params.interpolate_empty_values = False
    df, raw_df = load_data(params)

    Attributes:
        freq: A frequency string (must be in the pandas freq string format) to resample the data by.
        shuffle: bool, whether to shuffle data rows (the default is True).
        prev_load_values: int, how many previous load timestamps to include in each data row.
                            Use 0 to not include previous load values.
        prev_day_load_values: tuple[int, int], what window of previous day load timestamps to include in each row,
                            for example (-1, 1) means that the load on the previous day at the same timestamp
                            and for 2 neighbouring ones should be included in each data row.
                            Use (0, 0) to not include previous day load values.
        prev_load_as_mean: If True, the previous load values are aggreagted to a mean, if False, they are stored separately.
        prev_day_load_as_mean: If True, the previous day load values are aggregated to a mean, if False, they are stored separately.
        prev_temp_values: int, how many previous temperature timestamps to include in each data row.
                            Use 0 to not include previous temperature values.
        prev_day_temp_values: tuple[int, int], what window of previous day temperature timestamps to include in each row,
                            for example (-1, 1) means that the temperature on the previous day at the same timestamp
                            and for 2 neighbouring ones should be included in each data row.
                            Use (0, 0) to not include previous day temperature values.
        prev_temp_as_mean: If True, the previous temperature values are aggregated to a mean, if False, they are stored separately.
        prev_day_temp_as_mean: If True, the previous day temperature values are aggregated to a mean, if False, they are stored separately.
        interpolate_empty_values: bool, whether rows at the beginning of the DataFrame, that don't have previous values,
                            should be filled with mean values (True) or dropped (False).
    """
    freq: str = "1h"
    shuffle: bool = True
    prev_load_values: int = 3
    prev_day_load_values: tuple[int, int] = (-2, 2)
    prev_load_as_mean: bool = False
    prev_day_load_as_mean: bool = False
    prev_temp_values: int = 3
    prev_day_temp_values: tuple[int, int] = (-2, 2)
    prev_temp_as_mean: bool = True
    prev_day_temp_as_mean: bool = True
    interpolate_empty_values: bool = True
    

def load_training_data(params: DataLoadingParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Returns the training data (years 2015-2022) loaded and processed for machine learning,
    as well as the data with real values (which is meant to provide access to the real values in case
    they are needed during model evaluation, see load_data_raw for the raw data for data analysis).

    :param params: Configuration, see DataLoadingParams documentation.
    :returns: The DataFrame prepared for machine learning and a DataFrame with real values.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """

    return _load_data(params, TRAINING_YEARS)


def load_test_data(params: DataLoadingParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Returns the test data (years 2023-2024) loaded and processed for machine learning,
    as well as the data with real values (which is meant to provide access to the real values in case
    they are needed during model evaluation, see load_data_raw for the raw data for data analysis).

    :param params: Configuration, see DataLoadingParams documentation.
    :returns: The DataFrame prepared for machine learning and a DataFrame with real values.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """

    return _load_data(params, TEST_YEARS)


def decode_ml_outputs(to_decode: Tensor | pd.DataFrame, raw: pd.DataFrame):
    """
    Returns the power grid loads in MW corresponding to give scaled values.
    !!! IMPORTANT: raw has to be the same DataFrame returned by load_training_data
    (the raw DataFrame, NOT the ml-ready one). Otherwise the function will
    produce random results or throw an exception.
    
    :param to_decode: The NN's output or other standardized 'load' value to decode.
    :param raw: The *raw* DataFrame as returned by load_training_data (always
    the DataFrame from load_training_data, not load_test_data, as that was used to train the model).
    """
    scaler = StandardScaler()
    scaler.fit(raw[['load']])
    
    return scaler.inverse_transform(to_decode)


def load_raw_data(years: list[int], months: list[int]) -> pd.DataFrame:
    """ 
    Returns the loaded raw data in a format suitable for data analysis (the whole dataset, 2015-2024).

    :param years: List of years to load.
    :param months: List of months (0-11) to load (same for every specified year, to get
                    different months for each year call the function multiple times).
    :returns: The DataFrame.
    :rtype: pd.DataFrame
    """
    params = DataLoadingParams()
    params.freq = "15min"
    params.shuffle = False

    if pd.tseries.frequencies.to_offset(params.freq) < pd.tseries.frequencies.to_offset("15min"):
        raise ValueError("only resampling to lower frequencies is supported")

    df = pd.DataFrame()
    
    if any([y in TEST_YEARS for y in years]):
        raise ValueError("use of test data in analysis is not allowed")

    # load data for all selected years
    df = _load_from_raw(df, params, years)
    
    # drop the unnecessary 'forecast' column
    df = _drop_columns(df, params)
    
    # convert dates to correct type
    df = _get_dates(df, params)
    
    # rename columns
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "load"}, inplace=True)

    df = _get_temperature_raw(df, params, years)

    # keep only specified months
    df = _select_months(df, months)

    return df


#######
# internals
####### 

# do not modify, for model evaluation consistency
TRAINING_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_YEARS = [2023, 2024]

def _load_data(params, years):
    try:
        pd.tseries.frequencies.to_offset(params.freq)
    except ValueError as e:
        raise e
    
    if pd.tseries.frequencies.to_offset(params.freq) < pd.tseries.frequencies.to_offset("15min"):
        raise ValueError("only resampling to lower frequencies is supported")
    
    if params.prev_load_values < 0 or params.prev_load_values > 240 or params.prev_temp_values < 0 or params.prev_temp_values > 240:
        raise ValueError("allowed values for prev_load_values and prev_temp_values are in the range 0-240")
    
    if params.prev_day_load_values[0] > params.prev_day_load_values[1]:
        raise ValueError("prev_day_load_values should be a valid range-like tuple")
    
    if params.prev_day_temp_values[0] > params.prev_day_temp_values[1]:
        raise ValueError("prev_day_temp_values should be a valid range-like tuple")
    
    df = pd.DataFrame()

    # load data for all selected years
    df = _load_from_raw(df, params, years)
    
    # drop the unnecessary 'forecast' column
    df = _drop_columns(df, params)
    
    # convert dates to correct type
    df = _get_dates(df, params)
    
    # rename columns
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "load"}, inplace=True)
    
    # remove daylight savings related nans
    df = _remove_daylight_savings_nans(df, params, years)

    # resample data
    df = _resample(df, params)

    # add previous values
    df = _get_previous_loads(df, params)

    # add previous day values
    df = _get_previous_day_loads(df, params)

    # get day and hour numbers
    df = _get_date_numbers(df, params)
    
    # load temperature data
    df = _get_temperature(df, params, years)

    # add previous temperature values
    df = _get_previous_temps(df, params)

    # add previous day temperature values
    df = _get_previous_day_temps(df, params)

    # handle empty values at the beginning of the dataframe (they could not have previous values added)
    df = _handle_sliding_window_nans(df, params)

    # drop temporary 'temperature' column
    df = df.drop('temperature', axis=1)

    real_data_df = df.copy()
    if params.shuffle:
        real_data_df = shuffle(real_data_df, random_state=LOAD_DATA_RANDOM_STATE) # shuffle the 'real' df the same way as the ml-ready df

    # get ml-ready df
    df = _get_ml_ready_df(df, params, years == TRAINING_YEARS)

    return df, real_data_df



def _load_from_raw(df, params, years):
    for year in years:
        df = pd.concat(
            [df, pd.read_csv(Path(__file__).parent.parent.resolve()
                            / Path(f"data/raw/germany_{year}_15min.csv"))]
        )
    df.reset_index(drop=True, inplace=True)
    
    return df


def _drop_columns(df, params):
    df = df.iloc[:, [0, 2]]
    
    return df


def _get_dates(df, params):
    df[df.columns[0]] = df.iloc[:, 0].str.split(' - ').str[0]
    df[df.columns[0]] = pd.to_datetime(df.iloc[:, 0], format='%d.%m.%Y %H:%M')
    
    return df


def _remove_daylight_savings_nans(df, params, years):
    dst_start_dates = []
    for year in years:
        last_day_of_march = pd.to_datetime(f'{year}-03-31')
        
        dst_day = last_day_of_march - pd.Timedelta(days=(last_day_of_march.dayofweek - 6) % 7)
        dst_start_dates.append(dst_day.normalize())
    
    for date in dst_start_dates:
        start = date + pd.to_timedelta("02:00:00")
        end = date + pd.to_timedelta("02:59:59")
        
        df_temp = df.copy().set_index("date")
        df_temp.index = pd.to_datetime(df_temp.index)
        to_drop = df_temp.loc[df_temp.index.date == start.date()].between_time(start.time(), end.time(), inclusive='left').reset_index(drop=True)
        
        df_cleaned = df.merge(
            to_drop,
            how='left',
            indicator=True
        )
        df = df_cleaned[df_cleaned['_merge'] == 'left_only'].drop(columns=['_merge'])
        df = df.reset_index(drop=True)
        
    return df


def _get_date_numbers(df, params):
    df["day_of_week"] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['hour_of_day'] = df['date'].dt.hour
    
    return df


def _resample(df, params):
    df = df.set_index("date")
    df = df.resample(params.freq).sum()
    df = df.reset_index()

    return df


def _get_previous_loads(df, params):
    if params.prev_load_values <= 0:
        return df
    
    new_cols = []
    for i in range(1, params.prev_load_values + 1):
            df[f"load_timestamp_-{i}"] = df["load"].shift(i)
            new_cols.append(f"load_timestamp_-{i}")

    if params.prev_load_as_mean:
        df[f"prev_{params.prev_load_values}_timestamps_mean"] = df[new_cols].mean(axis=1)
        df = df.drop(columns=new_cols)

    return df


def _get_previous_day_loads(df, params):
    if params.prev_day_load_values == (0, 0):
        return df
    
    freq = df.set_index('date').index.inferred_freq
    freq = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    fit_count = int(pd.to_timedelta('1D') / freq)
    
    new_cols = []
    for i in range(params.prev_day_load_values[0], params.prev_day_load_values[1] + 1):
            df[f"load_previous_day_timestamp_{i}"] = df["load"].shift(i + fit_count)
            new_cols.append(f"load_previous_day_timestamp_{i}")

    if params.prev_day_load_as_mean:
        df[f"prev_day_load_{len(range(params.prev_day_load_values[0], params.prev_day_load_values[1] + 1))}_timestamps_mean"] = df[new_cols].mean(axis=1)
        df = df.drop(columns=new_cols)

    return df


def _get_temperature_raw(df, params, years):
    tdf = pd.read_csv(Path(__file__).parent.parent.resolve()
                            / Path(f"data/raw/germany_temperature_2015-2024.csv"))
    
    tdf['date'] = pd.to_datetime(tdf['date'])

    tdf = tdf[tdf['date'].dt.year.isin(years)]

    tdf = tdf.set_index("date")
    tdf = tdf.resample(params.freq).mean().ffill()
    tdf = tdf.reset_index()

    tdf = tdf.set_index('date')
    df = df.set_index('date')

    merged_df = pd.merge(
        df, 
        tdf, 
        left_index=True, 
        right_index=True, 
        how='left'
    )

    merged_df.reset_index()

    return merged_df


def _get_temperature(df, params, years):
    tdf = pd.read_csv(Path(__file__).parent.parent.resolve()
                            / Path(f"data/raw/germany_temperature_2015-2024.csv"))
    
    tdf['date'] = pd.to_datetime(tdf['date'])

    tdf = tdf[tdf['date'].dt.year.isin(years)]

    tdf = tdf.set_index("date")
    tdf = tdf.resample(params.freq).mean().interpolate(method='linear', limit_direction="both")
    tdf = tdf.reset_index()

    tdf = tdf.set_index('date')
    df = df.set_index('date')

    merged_df = pd.merge(
        df, 
        tdf, 
        left_index=True, 
        right_index=True, 
        how='left'
    )

    merged_df = merged_df.reset_index()
    merged_df['temperature'] = merged_df['temperature'].interpolate(method='linear', limit_direction="both")

    return merged_df


def _get_previous_temps(df, params):
    if params.prev_temp_values <= 0:
        return df
    
    new_cols = []
    for i in range(1, params.prev_temp_values + 1):
            df[f"temperature_timestamp_-{i}"] = df["temperature"].shift(i)
            new_cols.append(f"temperature_timestamp_-{i}")

    if params.prev_temp_as_mean:
        df[f"prev_{params.prev_temp_values}_temperature_timestamps_mean"] = df[new_cols].mean(axis=1)
        df = df.drop(columns=new_cols)

    return df


def _get_previous_day_temps(df, params):
    if params.prev_day_temp_values == (0, 0):
        return df
    
    freq = df.set_index('date').index.inferred_freq
    freq = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    fit_count = int(pd.to_timedelta('1D') / freq)
    
    new_cols = []
    for i in range(params.prev_day_temp_values[0], params.prev_day_temp_values[1] + 1):
            df[f"temperature_previous_day_timestamp_{i}"] = df["temperature"].shift(i + fit_count)
            new_cols.append(f"temperature_previous_day_timestamp_{i}")

    if params.prev_day_temp_as_mean:
        df[f"prev_day_temperature_{len(range(params.prev_day_temp_values[0], params.prev_day_temp_values[1] + 1))}_timestamps_mean"] = df[new_cols].mean(axis=1)
        df = df.drop(columns=new_cols)

    return df


def _handle_sliding_window_nans(df, params):
    if params.interpolate_empty_values:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='backward')

        df = df.reset_index()
    else:
        df = df.dropna()
    

    return df


def _get_ml_ready_df(df, params, is_training_data):
    # apply sine-cosine transformation for cyclical features
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / df['hour_of_day'].max())
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / df['hour_of_day'].max())

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / df['day_of_week'].max())
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / df['day_of_week'].max())

    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / df['day_of_year'].max())
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / df['day_of_year'].max())

    df.drop(columns=['day_of_year', 'hour_of_day', 'day_of_week'], inplace=True)
    df.set_index('date', inplace=True)

    # standardize other features
    scaler = StandardScaler()
    
    # this is done to ensure the data is always scaled according to the training data
    # it does not introduce data leakage, it emulates a model's pipeline
    if not is_training_data:
        _, training_df = load_training_data(params)
    else:
        training_df = df
    
    cols_to_scale = [col for col in df.columns if col not in 
                        ['hour_of_day_sin', 'hour_of_day_cos', 'day_of_week_sin', 'day_of_week_cos',
                            'day_of_year_sin', 'day_of_year_cos', 'date']]

    scaler.fit(training_df[cols_to_scale])
    scaled_values = scaler.transform(df[cols_to_scale])
    training_df = None

    df_scaled = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df.index)

    df_final = df.drop(cols_to_scale, axis=1)
    df_final = pd.concat([df_final, df_scaled], axis=1)
    
    # shuffle data rows
    if params.shuffle:
        df_final = shuffle(df_final, random_state=LOAD_DATA_RANDOM_STATE)

    return df_final


def _select_months(df, months):
    df = df.reset_index()

    mask = (df['date'].dt.month - 1).isin(months)

    df = df[mask]

    df = df.set_index('date')

    return df