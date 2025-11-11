import os
from pathlib import Path
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
        path (str): Directory path where plots will be saved. Default is "plots/input_data_plots"""

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

        self.path = Path("plots") / "input_data_plots"

        sns.set_theme(style='whitegrid')

    def create_line_chart_with_index(self, y: list[str], title: str = None, normalize: bool = False, save_plot: bool = False, folder: str = "line_plots", filename: str = "new_plot", show_plot: bool = False) -> None:
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
                Folder (relative to self.path) where the plot will be saved. Default is "line_plots".

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
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_line_charts_with_index(self, y: list[str], title: str = None, normalize: bool = False, save_plot: bool = False, folder: str = "line_plots", filename: str = "new_plot", show_plot: bool = False) -> None:
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
                Folder (relative to self.path) where the figure will be saved. Default is "line_plots".

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
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_correlation_matrix(self, save_plot: bool = False, title: str = None, folder: str = "correlation_matrices", filename: str = "new_correlation_matrix", show_plot: bool = False) -> None:
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
                Folder (relative to self.path) where the plot will be saved. Default is "correlation_matrices".

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
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_correlation_matrices(self, save_plot: bool = False, title: str = None, folder: str = "correlation_matrices", filename: str = "new_correlation_matrices", show_plot: bool = False) -> None:
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
                Folder (relative to self.path) where the figure will be saved. Default is "correlation_matrices".

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

                ax.set_title(f'{self.month_names[month]} {year}', fontsize=10)

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
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_bar_chart(self, function: Callable, feature: str, time_period: str = 'year', normalize: bool = False, save_plot: bool = False, folder: str = "bar_charts", filename: str = "new_bar_chart", show_plot: bool = False) -> None:
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
                Folder (relative to self.path) where the plot will be saved. Default is "bar_charts".

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

        elif time_period == 'month':
            aggregated.index = aggregated.index.map(lambda x: self.month_names[x - 1])
            order = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            aggregated.index = pd.CategoricalIndex(aggregated.index, categories=order, ordered=True)
            aggregated = aggregated.sort_index()

        if normalize:
            aggregated = self._normalize_data(aggregated)

        figure, axis = plt.subplots(figsize=(12, 6))
        aggregated.plot(kind='bar', ax=axis)
        axis.set_xlabel(time_period.capitalize().replace('_', ' '))
        axis.set_ylabel(f'{function.__name__.capitalize()} of {feature}')
        axis.set_title(f'{feature.capitalize()} aggregated by {time_period.replace("_", " ")}')
        plt.tight_layout()

        if save_plot:
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_qq_plot(self, x_feature: str, y_feature: str, quantile_count: int, title: str = None, save_plot: bool = False, folder: str = 'qq_plots', filename: str = 'new_qq_plot', show_plot: bool = False) -> None:
        """
        Creates a Quantile–Quantile (QQ) plot comparing two features.

        The method sorts the values of the specified features, divides them into the
        specified number of quantiles, and plots the quantiles against each other.
        This allows visualization of the relationship between the distributions of
        the two features. A reference line at 45 degrees is drawn to indicate perfect
        agreement between the distributions. The plot can be optionally saved to disk
        or displayed interactively.

        Parameters:
            x_feature (str):
                The name of the column to be used as the X-axis in the QQ plot.

            y_feature (str):
                The name of the column to be used as the Y-axis in the QQ plot.

            quantile_count (int):
                The number of quantiles to split each feature into for comparison.

            title (str, optional):
                Title of the plot. Default is None.

            save_plot (bool, optional):
                If True, saves the plot to disk. Default is False.

            folder (str, optional):
                Folder (relative to self.path) where the plot will be saved. Default is "qq_plots".

            filename (str, optional):
                Name of the file to save the plot. Default is "new_qq_plot".

            show_plot (bool, optional):
                If True, displays the plot interactively. Default is False.

        Returns:
            None
        """

        figure, axis = plt.subplots()

        x_data = self.raw_df[x_feature].dropna().sort_values().reset_index(drop=True)
        y_data = self.raw_df[y_feature].dropna().sort_values().reset_index(drop=True)

        x_data = self._normalize_data(x_data)
        y_data = self._normalize_data(y_data)

        x_quantiles = self._quantile_split(x_data, quantile_count)
        y_quantiles = self._quantile_split(y_data, quantile_count)

        min_val = min(x_quantiles.min(), y_quantiles.min())
        max_val = max(x_quantiles.max(), y_quantiles.max())
        plt.plot([max_val, min_val], [max_val, min_val], color='red', linestyle='--', label='Reference line (y=x)')
        plt.scatter(x_quantiles, y_quantiles)
        plt.xlabel(x_feature.capitalize())
        plt.ylabel(y_feature.capitalize())

        if title is not None:
            plt.title(title)
        plt.tight_layout()

        if save_plot:
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_auto_correlation_plot(self, feature: str, lags: int, include_confidence_bounds: bool = True, title: str = None, save_plot: bool = False, folder: str = 'autocorrelation_plots', filename: str = 'new_autocorrelation_plot', show_plot: bool = False) -> None:
        """
        Creates an autocorrelation plot for a selected feature over a specified number of lags.

        The method computes and visualizes autocorrelation values for the given feature to assess
        how current values relate to past observations at different lag values. Optionally, the
        plot can include 95% confidence bounds to help identify statistically significant
        autocorrelation levels. The resulting plot may be saved to disk or displayed interactively.

        Parameters
        ----------
        feature : str
            Name of the column from the dataset for which the autocorrelation should be computed.

        lags : int
            Number of lag values to include in the autocorrelation plot.

        include_confidence_bounds : bool, optional (default=True)
            If True, displays upper and lower 95% confidence bounds on the plot to help determine
            whether autocorrelation values are statistically significant.

        title : str, optional
            Title of the plot. If None, no title is displayed.

        save_plot : bool, optional (default=False)
            If True, saves the generated plot to disk.

        folder : str, optional (default='autocorrelation_plots')
            Folder (relative to self.path) where the plot will be saved.

        filename : str, optional (default='new_autocorrelation_plot')
            Name of the file to save the plot.

        show_plot : bool, optional (default=False)
            If True, displays the plot interactively.

        Returns
        -------
        None
        """

        figure, axis = plt.subplots()
        data_series = self.raw_df[feature].dropna()
        data_series = (data_series - data_series.mean()) / data_series.std()
        axis.acorr(data_series, maxlags=lags, usevlines=True)
        axis.set_xlim(0, lags)

        if include_confidence_bounds:
            confidence_bound = 1.96 / np.sqrt(len(data_series))
            axis.axhline(confidence_bound, color='blue', linestyle='--')
            axis.axhline(-confidence_bound, color='blue', linestyle='--')

        axis.set_xlabel('Lags')
        axis.set_ylabel(f'Autocorrelation of {feature}')

        if title is not None:
            axis.set_title(title)
        plt.tight_layout()

        if save_plot:
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_histogram(self, feature: str, bins: int = 30, title: str = None, save_plot: bool = False, folder: str = 'histograms', filename: str = 'new_histogram', show_plot: bool = False) -> None:
        """
        Creates a histogram for a selected feature from the dataset.

        The method visualizes the distribution of values for the specified feature using a histogram.
        The number of bins can be customized. Optionally, the plot can be titled, saved to disk,
        or displayed interactively.

        Parameters
        ----------
        feature : str
            Name of the column from the dataset for which the histogram should be generated.

        bins : int, optional (default=30)
            Number of bins to divide the feature values into.

        title : str, optional
            Title of the plot. If None, no title is displayed.

        save_plot : bool, optional (default=False)
            If True, saves the generated histogram to disk.

        folder : str, optional (default='histograms')
            Folder (relative to self.path) where the plot will be saved.

        filename : str, optional (default='new_histogram')
            Name of the file to save the histogram.

        show_plot : bool, optional (default=False)
            If True, displays the histogram interactively.

        Returns
        -------
        None
        """

        figure, axis = plt.subplots()
        axis.hist(self.raw_df[feature], bins=bins, edgecolor='black')
        axis.set_xlabel(f'{feature.capitalize()}')
        axis.set_ylabel('Number of observations')

        if title is not None:
            axis.set_title(title)
        plt.tight_layout()

        if save_plot:
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    def create_weekly_profile_heat_map(self, feature: str, function: str = 'mean', title: str = None, save_plot: bool = False, folder: str = 'weekly_heat_maps', filename: str = 'new_weekly_heat_map', show_plot: bool = False) -> None:
        """
        Creates a 24h × 7-day heatmap for a selected feature from the dataset.

        The method visualizes the distribution of a specified feature across hours of the day and days of the week.
        It aggregates values using the provided function (e.g., mean, sum) to show typical weekly patterns.
        Optionally, the plot can be titled, saved to disk, or displayed interactively.

        Parameters
        ----------
        feature : str
            Name of the column from the dataset to visualize in the heatmap.

        function : str, optional (default='mean')
            Aggregation function to apply when multiple values exist for the same hour and day combination.
            Supported options include 'mean', 'sum', 'max', 'min', etc.

        title : str, optional
            Title of the plot. If None, no title is displayed.

        save_plot : bool, optional (default=False)
            If True, saves the generated heatmap to disk.

        folder : str, optional (default='heatmaps')
            Folder (relative to self.path) where the plot will be saved.

        filename : str, optional (default='new_heatmap')
            Name of the file to save the heatmap.

        show_plot : bool, optional (default=False)
            If True, displays the heatmap interactively.

        Returns
        -------
        None
        """

        df = self.raw_df.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.day_name()

        heatmap_data = df.pivot_table(index='day', columns='hour', values=feature, aggfunc=function)

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)


        figure, axis = plt.subplots()
        sns.heatmap(heatmap_data, ax=axis)

        if title is not None:
            axis.set_title(title)
        plt.tight_layout()

        if save_plot:
            save_dir = os.path.join(self.path, folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}.png')
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close(figure)

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

    def _quantile_split(self, data: np.array, quantile_count: int) -> np.array:
        """
        Splits a sorted array or Pandas Series into a specified number of quantiles.

        The method calculates the positions corresponding to the requested number of
        quantiles and extracts the values at these positions. This is used for
        creating QQ plots or other quantile-based analyses. Raises an error if the
        number of quantiles exceeds the number of data points.

        Parameters:
            data (np.array or pd.Series):
                The data array or Pandas Series to be split into quantiles. It should
                be sorted and may have missing values removed before calling this method.

            quantile_count (int):
                The number of quantiles to split the data into. Must be less than or
                equal to the number of data points.

        Returns:
            np.array:
                Array containing the values at the calculated quantile positions.
        """

        data_point_count = len(data)
        if quantile_count > data_point_count:
            raise ValueError('The number of quantiles cannot exceed the number of data points.')

        indices = np.linspace(0, data_point_count - 1, quantile_count, dtype=int)
        quantile_values = data.iloc[indices]

        return np.array(quantile_values)
