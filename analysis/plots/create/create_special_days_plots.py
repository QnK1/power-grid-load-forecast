import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.load_data import load_raw_data, TRAINING_YEARS
from utils.get_easter_days import get_easter_days_for_years


def plot_top_outliers(k: int):
    YEARS = TRAINING_YEARS
    
    df = load_raw_data(
        YEARS,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    
    # different means are calculated in order to compare potential outliers to them
    
    df = df.reset_index()
    print(df)
    
    total_mean = df['load'].mean()
    
    df['dayofweek'] = df['date'].dt.day_of_week
    mean_by_dayofweek = df.groupby('dayofweek')['load'].mean()
    
    df["dayofmonth"] = df['date'].dt.day
    mean_by_dayofmonth = df.groupby('dayofmonth')['load'].mean()
    
    
    df['total_diff'] = (np.abs(df['load'] - total_mean) / total_mean) * 100
    df['dayofweek_diff'] = (np.abs(df['load'] - df['dayofweek'].map(mean_by_dayofweek)) / df['dayofweek'].map(mean_by_dayofweek)) * 100
    df['dayofmonth_diff'] = (np.abs(df['load'] - df['dayofmonth'].map(mean_by_dayofmonth)) / df['dayofmonth'].map(mean_by_dayofmonth)) * 100
    
    mean_total_diff = df.groupby([df['date'].dt.month, df['date'].dt.day])['total_diff'].mean().rename_axis(['month', 'day'])
    mean_dayofweek_diff = df.groupby([df['date'].dt.month, df['date'].dt.day])['dayofweek_diff'].mean().rename_axis(['month', 'day'])
    mean_dayofmonth_diff = df.groupby([df['date'].dt.month, df['date'].dt.day])['dayofmonth_diff'].mean().rename_axis(['month', 'day'])
    
    mean_total_diff = mean_total_diff.reset_index()
    mean_dayofweek_diff = mean_dayofweek_diff.reset_index()
    mean_dayofmonth_diff = mean_dayofmonth_diff.reset_index()
    
    mean_total_diff.sort_values(by='total_diff', inplace=True, ascending=False)
    mean_dayofweek_diff.sort_values(by='dayofweek_diff', inplace=True, ascending=False)
    mean_dayofmonth_diff.sort_values(by='dayofmonth_diff', inplace=True, ascending=False)
    
    mean_total_diff['date_str'] = mean_total_diff['day'].astype(str) + "." + mean_total_diff['month'].astype(str)
    mean_dayofweek_diff['date_str'] = mean_dayofweek_diff['day'].astype(str) + "." + mean_dayofweek_diff['month'].astype(str)
    mean_dayofmonth_diff['date_str'] = mean_dayofmonth_diff['day'].astype(str) + "." + mean_dayofmonth_diff['month'].astype(str)
    
    # create plots
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(14, 6))
    fig.suptitle('Top 20 Days With Load Furthest From Respective Mean', fontsize=18)
    ax[0].set_title('Difference From Total Mean')
    handle = ax[0].axvline(x=mean_total_diff['total_diff'].median(), color='r', linestyle='--', linewidth=2, label='Median Difference')
    ax[1].set_title('Difference From Mean For Given Day Of Week')
    ax[1].axvline(x=mean_dayofweek_diff['dayofweek_diff'].median(), color='r', linestyle='--', linewidth=2, label='Median Difference')
    ax[2].set_title('Difference From Mean For Given Day Of Month')
    ax[2].axvline(x=mean_dayofmonth_diff['dayofmonth_diff'].median(), color='r', linestyle='--', linewidth=2, label='Median Difference')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace=1)
    
    mean_total_diff.head(k).plot(
        ax=ax[0],
        kind="barh",
        x="date_str",
        y="total_diff",
        ylabel="",
        legend=False,
    )
    
    mean_dayofweek_diff.head(k).plot(
        ax=ax[1],
        kind="barh",
        x="date_str",
        y="dayofweek_diff",
        ylabel="",
        legend=False,
    )
    
    mean_dayofmonth_diff.head(k).plot(
        ax=ax[2],
        kind="barh",
        x="date_str",
        y="dayofmonth_diff",
        ylabel="",
        legend=False,
    )
    plt.tight_layout()
    fig.supxlabel('Difference From Mean [%]', fontsize=14)
    fig.supylabel('Date [D.M]', fontsize=14)
    fig.subplots_adjust(left=0.1, bottom=0.1)
    fig.legend(handles=[handle], loc='upper left', ncol=3, fontsize=12)
    plt.savefig(f"special_days.png")
    
    
    # handle easter days separately
    
    easter_dict = get_easter_days_for_years(TRAINING_YEARS)
    
    print(mean_total_diff)
    easter_total_diffs = {}
    for easter_day, day_list in easter_dict.items():
        date_tuples = pd.MultiIndex.from_arrays([mean_total_diff['day'], mean_total_diff['month']])
        mask = date_tuples.isin([(d.day, d.month) for d in day_list])
        filtered = mean_total_diff[mask]
        easter_total_diffs[easter_day] = [filtered['total_diff'].mean()]
    easter_total_diffs = pd.DataFrame(easter_total_diffs)
    
    easter_dayofweek_diffs = {}
    for easter_day, day_list in easter_dict.items():
        date_tuples = pd.MultiIndex.from_arrays([mean_dayofweek_diff['day'], mean_dayofweek_diff['month']])
        mask = date_tuples.isin([(d.day, d.month) for d in day_list])
        filtered = mean_dayofweek_diff[mask]
        easter_dayofweek_diffs[easter_day] = [filtered['dayofweek_diff'].mean()]
    easter_dayofweek_diffs = pd.DataFrame(easter_dayofweek_diffs)
    
    easter_dayofmonth_diffs = {}
    for easter_day, day_list in easter_dict.items():
        date_tuples = pd.MultiIndex.from_arrays([mean_dayofmonth_diff['day'], mean_dayofmonth_diff['month']])
        mask = date_tuples.isin([(d.day, d.month) for d in day_list])
        filtered = mean_dayofmonth_diff[mask]
        easter_dayofmonth_diffs[easter_day] = [filtered['dayofmonth_diff'].mean()]
    easter_dayofmonth_diffs = pd.DataFrame(easter_dayofmonth_diffs)
        
    
    # create plots
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(16, 9))
    fig.suptitle('Average Difference From Respective Mean For Easter-related Days', fontsize=18)
    ax[0].set_title('Average Difference From Total Mean')
    ax[1].set_title('Average Difference From Mean For Given Day Of Week')
    ax[2].set_title('Average Difference From Mean For Given Day Of Month')
    ax[0].yaxis.set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    # fig.subplots_adjust(hspace=1)
    
    easter_total_diffs.head(k).plot(
        ax=ax[0],
        kind="bar",
        ylabel="",
        legend=False,
    )
    
    easter_dayofweek_diffs.head(k).plot(
        ax=ax[1],
        kind="bar",
        ylabel="",
        legend=False,
    )
    
    easter_dayofmonth_diffs.head(k).plot(
        ax=ax[2],
        kind="bar",
        ylabel="",
        legend=False,
    )
    # plt.tight_layout()
    fig.supxlabel('Day', fontsize=14)
    fig.supylabel('Difference From Mean [%]', fontsize=14)
    fig.subplots_adjust(left=0.1, bottom=0.1)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', ncol=3)
    plt.savefig(f"special_days_easter.png")
    
if __name__ == "__main__":
    plot_top_outliers(20)