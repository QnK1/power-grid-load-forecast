from utils.load_data import load_raw_data
from matplotlib import pyplot as plt

class PlotCreator:
    def __init__(self, years: list[int], months: list[int]):
        self.raw_df = load_raw_data(years, months)
        self.path = "plots/"

    def create_line_charts(self, x: list[str], y: list[str], save_plot: bool = False, filename: str = "new_plot"):
        if len(x) != len(y):
            raise ValueError(f"Amount of parameters in x ({len(x)}) should be equal to the amount of parameters in y ({len(y)}).")

        figure, axis = plt.subplots(1, len(x))

        for i in range(len(x)):
            axis[i].plot(self.raw_df[x[i]], self.raw_df[y[i]])
            axis[i].set_xlabel(x[i])
            axis[i].set_ylabel(y[i])
            axis[i].tick_params(axis='x', rotation=90)
        figure.tight_layout()
        if save_plot:
            figure.savefig(f"{self.path}{filename}")
        plt.show()

# ----- TESTING -----
# data_params = DataLoadingParams()
# data_params.years = [2015]
# data_params.freq = "15min"

plot_creator = PlotCreator([2015], [3])
data = load_raw_data()
print(data.columns)
print(data.head())
#plot_creator.create_line_charts(["temperature"], ["load"])