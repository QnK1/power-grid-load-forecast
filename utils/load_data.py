import pandas as pd
from pathlib import Path

class DataLoadingParams:
    country: str = "germany"
    years: list[str] = ["2015", "2016", "2017", "2018", "2019"]
    original_freq: str = "15min"
    freq: str = "1h"
    prev_load_values: int = 3
    prev_day_load_values: tuple[int, int] = (-2, 2)
    prev_load_as_mean: bool = False
    prev_day_load_as_mean: bool = False
    prev_temp_values: int = 3
    prev_day_temp_values: tuple[int, int] = (-2, 2)
    prev_temp_as_mean: bool = True
    prev_day_temp_as_mean: bool = True
    
    
def load_data(params: DataLoadingParams) -> pd.DataFrame:
    df = pd.DataFrame()
    
    # load data for all selected years
    df = _load_from_raw(df, params)
    
    # drop the unnecessary 'forecast' column
    df = _drop_columns(df, params)
    
    # convert dates to correct type
    df = _get_dates(df, params)
    
    # rename columns
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "load"}, inplace=True)
    
    # remove daylight savings related nans
    df = _remove_daylight_savings_nans(df, params)
        
    # get day and hour numbers
    df = _get_date_numbers(df, params)
    
    print(df)


def _load_from_raw(df, params):
    for year in params.years:
        df = pd.concat(
            [df, pd.read_csv(Path(__file__).parent.parent.resolve()
                            / Path(f"data/raw/{params.country}_{year}_{params.original_freq}.csv"))]
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


def _remove_daylight_savings_nans(df, params):
    dst_start_dates = []
    for year in params.years:
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



if __name__ == "__main__":
    load_data(DataLoadingParams())