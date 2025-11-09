import numpy as np

from analysis.input_data_analysis import InputDataPlotCreator

#plot_creator = InputDataPlotCreator([2015], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# plot_creator.create_line_chart_with_index(["load", "temperature"], normalize=True, save_plot=False, show_plot=True)
# plot_creator.create_correlation_matrix(show_plot=True, title="July 2015 - 2019")
# plot_creator.create_correlation_matrices(show_plot=True)
# plot_creator.create_bar_chart(np.mean, "temperature", "day_of_week", normalize=False, save_plot=False, show_plot=True)
# plot_creator.create_bar_chart(sum, "load", normalize=False, save_plot=False, show_plot=True)
#plot_creator.create_qq_plot('load', 'temperature', 100, show_plot=True, save_plot=True)
#plot_creator.create_auto_correlation_plot('load', 960, show_plot=True, save_plot=True)
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
months = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

month_names = {
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

for year in years:
    for month in months:
        plot_creator = InputDataPlotCreator([year], [month])
        plot_creator.create_line_chart_with_index(["load", "temperature"], title=f"{month_names[month]} {year}", filename=f"{month_names[month]}{year}", normalize=True, save_plot=True, show_plot=False)