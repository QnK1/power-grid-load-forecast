import numpy as np

from analysis.input_data_analysis import InputDataPlotCreator

plot_creator = InputDataPlotCreator([2015, 2016, 2017, 2018, 2019], [6])
plot_creator.create_line_chart_with_index(["load", "temperature"], normalize=True, save_plot=False, show_plot=True)
plot_creator.create_correlation_matrix(show_plot=True, title="July 2015 - 2019")
plot_creator.create_correlation_matrices(show_plot=True)
plot_creator.create_bar_chart(np.mean, "temperature", "day_of_week", normalize=False, save_plot=False, show_plot=True)
plot_creator.create_bar_chart(sum, "load", normalize=False, save_plot=False, show_plot=True)