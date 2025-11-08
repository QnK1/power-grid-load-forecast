from typing import Callable
from utils.load_data import load_raw_data
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class InputDataPlotCreator:
    """
    Class for creating line charts from a raw DataFrame using the index as the X-axis.

    This class allows plotting one or multiple columns of a DataFrame, with optional
    normalization of values to the 0–1 range, rotation of X-axis labels for readability,
    and saving plots to a specified directory.

    Example usage:
        plot_creator = PlotCreator(years=[2023, 2024], months=[1, 2, 3])
        plot_creator.create_line_chart_with_index(["load"], normalize=True, save_plot=True, filename="load_plot")
        plot_creator.create_line_charts_with_index(["temperature", "load"], normalize=False)

    Attributes:
        raw_df (pd.DataFrame): The raw data loaded for the specified years and months.
        years (list[int]): List of years included in the dataset and used for plotting.
        months (list[int]): List of months included in the dataset and used for filtering plots.
        month_names (dict[int, str]): Mapping of month indices (0–11) to their English names.
        path (str): Directory path where plots will be saved. Default is "plots/"""

    def __init__(self, years: list[int], months: list[int]):
        """
        Initializes the PlotCreator instance by loading the raw data for given years and months.

        :param years: List of years to load.
        :param months: List of months to load.
        """

        self.raw_df = load_raw_data(years, months)
        self.months = months
        self.years = years

        self.month_names = {
            0: 'January',
            1: 'February',
            2: 'March',
            3: 'April',
            4: 'May',
            5: 'June',
            6: 'July',
            7: 'August',
            8: 'September',
            9: 'October',
            10: 'November',
            11: 'December'
        }

        self.path = 'plots/input_data_plots/'

        sns.set_theme(style='whitegrid')

    def create_line_chart_with_index(self, y: list[str], title: str = None, normalize: bool = False, save_plot: bool = False, folder: str = "line_plots/", filename: str = "new_plot", show_plot: bool = False) -> None:
        """
        Plots one or more columns on a single line chart using the DataFrame index as X-axis.

        Optional normalization scales values to the 0–1 range. Automatically rotates X-axis
        labels for readability and adds a legend with column names. Can save the plot to disk.

        Parameters:
            y (list[str]):
                List of column names to plot.

            title (str, optional):
                Title to display on the plot. If None, no title is set. Default is None.

            normalize (bool, optional):
                If True, scales the values of each column to the 0–1 range. Default is False.

            save_plot (bool, optional):
                If True, saves the plot to disk. Default is False.

            folder (str, optional):
                Folder (relative to self.path) where the plot will be saved. Default is "line_plots/".

            filename (str, optional):
                Name of the file to save the plot. Default is "new_plot".

            show_plot (bool, optional):
                If True, displays the plot interactively. Default is False.

        Returns:
            None
        """

        figure, axis = plt.subplots()
        for col in y:
            data = self._normalize_data(self.raw_df[col]) if normalize else self.raw_df[col]
            axis.plot(self.raw_df.index, data, label=col)
        axis.set_xlabel(self.raw_df.index.name)
        axis.set_ylabel('Normalized Value' if normalize else 'Value')
        axis.legend()
        plt.xticks(rotation=90)

        if title is not None:
            plt.title(title)
        plt.tight_layout()

        if save_plot:
            figure.savefig(f'{self.path}{folder}{filename}')
        if show_plot:
            plt.show()
        plt.close(figure)

    def create_line_charts_with_index(self, y: list[str], title: str = None, normalize: bool = False, save_plot: bool = False, folder: str = "line_plots/", filename: str = "new_plot", show_plot: bool = False) -> None:
        """
        Plots multiple subplots in a single row, one subplot per column in `y`.

        Each subplot uses the DataFrame index as X-axis and includes optional normalization.
        Automatically rotates X-axis labels and sets Y-axis labels to the column names.
        Can save the figure to disk.

        Parameters:
            y (list[str]):
                List of column names to plot.

            title (str, optional):
                Title to display on the plot. If None, no title is set. Default is None.

            normalize (bool, optional):
                If True, scales the values of each column to the 0–1 range. Default is False.

            save_plot (bool, optional):
                If True, saves the figure to disk. Default is False.

            folder (str, optional):
                Folder (relative to self.path) where the figure will be saved. Default is "line_plots/".

            filename (str, optional):
                Name of the file to save the figure. Default is "new_plot".

            show_plot (bool, optional):
                If True, displays the figure interactively. Default is False.

        Returns:
            None
        """

        figure, axis = plt.subplots(1, len(y))

        if len(y) == 1:
            axis = [axis]

        for i in range(len(y)):
            column = self._normalize_data(self.raw_df[y[i]]) if normalize else self.raw_df[y[i]]
            axis[i].plot(self.raw_df.index, column)
            axis[i].set_xlabel(self.raw_df.index.name)
            axis[i].set_ylabel(y[i])
            axis[i].tick_params(axis='x', rotation=90)

        if title is not None:
            plt.title(title)
        figure.tight_layout()

        if save_plot:
            figure.savefig(f'{self.path}{folder}{filename}')

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_correlation_matrix(self, save_plot: bool = False, title: str = None, folder: str = "correlation_matrices/", filename: str = "new_correlation_matrix", show_plot: bool = False) -> None:
        """
        Generates and displays a correlation heatmap for all numeric columns in the raw DataFrame.

        Computes pairwise correlation between all numerical features and visualizes them
        using a single color-coded heatmap via Seaborn. Optionally saves the plot to disk.

        Parameters:
            save_plot (bool, optional):
                If True, saves the heatmap as a PNG file. Default is False.

            title (str, optional):
                Title to display on the plot. If None, no title is set. Default is None.

            folder (str, optional):
                Folder (relative to self.path) where the plot will be saved. Default is "correlation_matrices/".

            filename (str, optional):
                Filename (without extension) to save the heatmap. Default is "new_correlation_matrix".

            show_plot (bool, optional):
                If True, displays the heatmap interactively. Default is False.

        Returns:
            None
        """

        figure, axis = plt.subplots()
        sns.heatmap(self.raw_df.corr(), annot=True, cmap="coolwarm", ax=axis)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        if save_plot:
            figure.savefig(f'{self.path}{folder}{filename}')
        if show_plot:
            plt.show()
        plt.close(figure)

    def create_correlation_matrices(self, save_plot: bool = False, title: str = None, folder: str = "correlation_matrices/", filename: str = "new_correlation_matrices", show_plot: bool = False) -> None:
        """
        Generates and displays a grid of correlation heatmaps for each month and year combination.

        For each year in `self.years` and month in `self.months`, computes pairwise correlations
        of numerical columns for that month, and displays them in a grid of heatmaps.
        Optionally saves the figure to disk.

        Parameters:
            save_plot (bool, optional):
                If True, saves the figure as a PNG file. Default is False.

            title (str, optional):
                Title to display on the plot. If None, no title is set. Default is None.

            folder (str, optional):
                Folder (relative to self.path) where the figure will be saved. Default is "correlation_matrices/".

            filename (str, optional):
                Filename (without extension) to save the figure. Default is "new_correlation_matrices".

            show_plot (bool, optional):
                If True, displays the figure interactively. Default is False.

        Returns:
            None
        """

        month_count, year_count = len(self.months), len(self.years)
        figure, axis = plt.subplots(month_count, year_count)

        for i, year in enumerate(self.years):
            for j, month in enumerate(self.months):
                if month_count == 1:
                    ax = axis[i]
                elif year_count == 1:
                    ax = axis[j]
                else:
                    ax = axis[j, i]

                sns.heatmap(self.raw_df[(self.raw_df.index.year == year) & (self.raw_df.index.month == month + 1)].corr(), annot=True, cmap="coolwarm", ax=ax, cbar=(i == year_count - 1))

                ax.set_title(f"{self.month_names[month]} {year}", fontsize=10)

                if j < month_count - 1:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                if i > 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

        if title is not None:
            plt.title(title)
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{self.path}{folder}{filename}")

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_bar_chart(self, function: Callable, feature: str, time_period: str = 'year', normalize: bool = False, save_plot: bool = False, folder: str = "bar_charts/", filename: str = "new_bar_chart", show_plot: bool = False) -> None:
        """
        Creates a bar chart of an aggregated feature grouped by years or months using a specified function.

        The method groups the data by a chosen time period (year, month, day, hour, or day_of_week), applies
        the given aggregation function to the selected feature, and visualizes the result in a bar chart.
        Optional normalization can be applied to scale aggregated values to the 0–1 range. The plot can be saved
        to disk or displayed interactively.

        Parameters:
            function (Callable):
                A function (e.g., mean, sum, max) used to aggregate the selected feature.

            feature (str):
                The name of the column to aggregate and visualize.

            time_period (str, optional):
                Defines the grouping level for aggregation. Must be one of:
                'year', 'month', 'day', 'hour', or 'day_of_week'. Default is 'year'.

            normalize (bool, optional):
                If True, applies min–max normalization (0–1) to aggregated values before plotting.
                Default is False.

            save_plot (bool, optional):
                If True, saves the plot to disk. Default is False.

            folder (str, optional):
                Folder (relative to self.path) where the plot will be saved. Default is "bar_charts/".

            filename (str, optional):
                Name of the file to save the plot. Default is "new_bar_chart".

            show_plot (bool, optional):
                If True, displays the plot interactively. Default is False.

        Returns:
            None
        """

        group_keys = {
            'year': self.raw_df.index.year,
            'month': self.raw_df.index.month,
            'day': self.raw_df.index.day,
            'hour': self.raw_df.index.hour,
            'day_of_week': self.raw_df.index.day_name()
        }

        if time_period not in group_keys.keys():
            raise ValueError(f'Given time_period is incorrect: It shoulde be one of: year, month, day, hour, day_of_week, but was {time_period} instead.')

        group_key =  group_keys[time_period]
        aggregated = self.raw_df.groupby(group_key)[feature].apply(function)

        if time_period == 'day_of_week':
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            aggregated.index = pd.CategoricalIndex(aggregated.index, categories=order, ordered=True)
            aggregated = aggregated.sort_index()

        if normalize:
            aggregated = self._normalize_data(aggregated)

        fig, ax = plt.subplots(figsize=(12, 6))
        aggregated.plot(kind="bar", ax=ax)
        ax.set_xlabel(time_period.capitalize().replace("_", " "))
        ax.set_ylabel(f'{function.__name__.capitalize()} of {feature}')
        ax.set_title(f'{feature.capitalize()} aggregated by {time_period.replace("_", " ")}')
        plt.tight_layout()

        if save_plot:
            fig.savefig(f'{self.path}{folder}{filename}')
        if show_plot:
            plt.show()
        plt.close(fig)

    def _normalize_data(self, column: pd.Series) -> pd.Series:
        """
        Normalizes a Pandas Series to the 0–1 range using min-max scaling.

        Parameters:
            column (pd.Series):
                The Pandas Series to normalize.

        Returns:
            pd.Series:
                A Pandas Series where the minimum value is 0 and the maximum value is 1.
        """

        return (column - column.min()) / (column.max() - column.min())
