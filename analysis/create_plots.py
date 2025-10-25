from utils.load_data import load_raw_data
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

class PlotCreator:
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
            0: "January",
            1: "February",
            2: "March",
            3: "April",
            4: "May",
            5: "June",
            6: "July",
            7: "August",
            8: "September",
            9: "October",
            10: "November",
            11: "December"
        }

        self.path = "plots/"

    def create_line_chart_with_index(self, y: list[str], normalize: bool = False, save_plot: bool = False, filename: str = "new_plot") -> None:
        """
        Plots one or more columns on a single line chart using the DataFrame index as X-axis.

        Optional normalization scales values to the 0–1 range. Automatically rotates X-axis
        labels for readability and adds a legend with column names. Can save the plot to disk.

        :param y: List of column names to plot.
        :param normalize: If True, scales the values of each column to the 0–1 range. Default is False.
        :param save_plot: If True, saves the plot to disk at self.path with the given filename. Default is False.
        :param filename: Name of the file to save the plot. Default is "new_plot".
        """

        for i in range(len(y)):
            column = self._normalize_data(self.raw_df[y[i]]) if normalize else self.raw_df[y[i]]
            plt.plot(self.raw_df.index, column, label=y[i])
        plt.xlabel(self.raw_df.index.name)
        plt.tick_params(axis='x', rotation=90)
        plt.tight_layout()
        plt.legend()
        if save_plot:
            plt.savefig(f"{self.path}{filename}")
        plt.show()

    def create_line_charts_with_index(self, y: list[str], normalize: bool = False, save_plot: bool = False, filename: str = "new_plot") -> None:
        """
        Plots multiple subplots in a single row, one subplot per column in `y`.

        Each subplot uses the DataFrame index as X-axis and includes optional normalization.
        Automatically rotates X-axis labels and sets Y-axis labels to the column names.
        Can save the figure to disk.

        :param y: List of column names to plot.
        :param normalize: If True, scales the values of each column to the 0–1 range. Default is False.
        :param save_plot: If True, saves the figure to disk at self.path with the given filename. Default is False.
        :param filename: Name of the file to save the figure. Default is "new_plot".
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
        figure.tight_layout()
        if save_plot:
            figure.savefig(f"{self.path}{filename}")
        plt.show()

    def create_correlation_matrix(self, save_plot: bool = False, filename: str = "new_correlation_matrix") -> None:
        """
        Generates and displays a correlation heatmap for all numeric columns in the raw DataFrame.

        This method computes the pairwise correlation between all numerical features in the dataset
        and visualizes the results as a color-coded heatmap using Seaborn. It can optionally save
        the plot as an image file in the directory specified by the `path` attribute.

        :param save_plot: If True, saves the plot as a PNG file to the configured directory. Default is False.
        :param filename: The filename (without extension) to use when saving the plot. Default is "new_correlation_matrix".
        :returns: None
        """

        sns.heatmap(self.raw_df.corr(), annot=True, cmap="coolwarm")
        if save_plot:
            plt.savefig(f"{self.path}{filename}")
        plt.show()

    def create_correlation_matrices_for_months(self, save_plot: bool = False, filename: str = "new_correlation_matrices") -> None:
        figure, axis = plt.subplots(len(self.months), len(self.years))

        for i, year in enumerate(self.years):
            for j, month in enumerate(self.months):
                sns.heatmap(self.raw_df[(self.raw_df.index.year == year) & (self.raw_df.index.month == month + 1)].corr(), annot=True, cmap="coolwarm", ax=axis[j, i], cbar=(i == len(self.years) - 1))

                axis[j, i].set_title(f"{self.month_names[month]} {year}", fontsize=10)

                if j < len(self.months) - 1:
                    axis[j, i].set_xticklabels([])
                    axis[j, i].set_xlabel("")
                if i > 0:
                    axis[j, i].set_yticklabels([])
                    axis[j, i].set_ylabel("")

        plt.tight_layout()
        if save_plot:
            plt.savefig(f"{self.path}{filename}")
        plt.show()


    def _normalize_data(self, column: pd.Series) -> pd.Series:
        """
        Normalizes a Pandas Series to the 0–1 range using min-max scaling.

        :param column: The Pandas Series to normalize.
        :returns: A Pandas Series where the minimum value is 0 and the maximum value is 1.
        """

        return (column - column.min()) / (column.max() - column.min())

# ----- TESTING -----

plot_creator = PlotCreator([2015, 2016, 2017, 2018], [4, 5, 6, 7, 8])
#plot_creator.create_line_chart_with_index(["load", "temperature"], True)
#plot_creator.create_correlation_matrix()
plot_creator.create_correlation_matrices_for_months()