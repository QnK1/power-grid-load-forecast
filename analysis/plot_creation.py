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
months = [6, 7, 8, 9, 10, 11] #

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

# for year in years:
#     for month in months:
#         plot_creator = InputDataPlotCreator([year], [month])
#         plot_creator.create_line_chart_with_index(["load", "temperature"], title=f"{month_names[month]} {year}", filename=f"{month_names[month]}{year}", normalize=True, save_plot=True, show_plot=False)

#plot_creator = InputDataPlotCreator(years, months)
#plot_creator.create_auto_correlation_plot('load', 96 * 7, title='Load autocorrelation', filename='load_autocorrelation', save_plot=True, include_confidence_bounds=False)

# Load
#plot_creator.create_bar_chart(sum, 'load', 'year', save_plot=True, filename='load_over_year_sum_bar_chart')
#plot_creator.create_bar_chart(sum, 'load', 'year', save_plot=True, filename='load_over_year_sum_bar_chart_normalized', normalize=True)
#plot_creator.create_bar_chart(sum, 'load', 'day_of_week', save_plot=True, filename='load_over_day_of_week_sum_bar_chart')
#plot_creator.create_bar_chart(np.mean, 'load', 'day_of_week', save_plot=True, filename='load_over_day_of_week_mean_bar_chart')
#plot_creator.create_bar_chart(np.mean, 'load', 'month', save_plot=True, filename='load_over_month_mean_bar_chart')
#plot_creator.create_bar_chart(np.mean, 'load', 'year', save_plot=True, filename='load_over_year_mean_bar_chart')

# Temperature
#plot_creator.create_bar_chart(np.mean, 'temperature', 'day_of_week', save_plot=True, filename='temperature_over_day_of_week_mean_bar_chart')
#plot_creator.create_bar_chart(np.mean, 'temperature', 'month', save_plot=True, filename='temperature_over_month_mean_bar_chart')
#plot_creator.create_bar_chart(np.mean, 'temperature', 'year', save_plot=True, filename='temperature_over_year_mean_bar_chart')

# for year in years:
#     plot_creator = InputDataPlotCreator([year], months)
#     plot_creator.create_correlation_matrix(save_plot=True, filename=f'{year}_load_temp_correlation', title=f'Correlation for {year}')
#
# for month in months:
#     plot_creator = InputDataPlotCreator(years, [month])
#     plot_creator.create_correlation_matrix(save_plot=True, filename=f'{month_names[month]}_load_temp_correlation', title=f'Correlation for every {month_names[month]} combined')

for year in years:
    plot_creator = InputDataPlotCreator([year], months)
    plot_creator.create_correlation_matrices(save_plot=True, filename=f'{year}_second_half_correlation')
