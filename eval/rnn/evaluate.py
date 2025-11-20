from eval.rnn.rnn import model_evaluate
from analysis.model_analysis import ModelPlotCreator

file_name_rnn_32_15 = 'RNN_model_NEURONS-32_EPOCHS-15_SEQUENCE-48_FREQ-1h'
file_name_rnn_32_20 = 'RNN_model_NEURONS-32_EPOCHS-20_SEQUENCE-48_FREQ-1h'
file_name_rnn_64_15 = 'RNN_model_NEURONS-64_EPOCHS-15_SEQUENCE-48_FREQ-1h'
file_name_rnn_64_20 = 'RNN_model_NEURONS-64_EPOCHS-20_SEQUENCE-48_FREQ-1h'


y_pred, y_true, mape_per_step_32_15 = model_evaluate(48, 24, file_name=file_name_rnn_32_15)
y_pred20, y_true20, mape_per_step_32_20 = model_evaluate(48, 24, file_name=file_name_rnn_32_20)
y_pred_64_15, y_true_64_15, mape_per_step_64_15 = model_evaluate(48, 24, file_name=file_name_rnn_64_15)
y_pred_64_20, y_true_64_20, mape_per_step_64_20 = model_evaluate(48, 24, file_name=file_name_rnn_64_20)


plot_creator = ModelPlotCreator()

plot_creator.plot_predictions(y_true[1000], y_pred[1000], file_name_rnn_32_15, 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true20[1000], y_pred20[15], file_name_rnn_32_20, 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true_64_15[1000], y_pred_64_15[1000], file_name_rnn_64_15, 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true_64_20[1000], y_pred_64_20[1000], file_name_rnn_64_20, 'rnn', freq='1h', show_plot=True)
# plot_creator.plot_mape_over_horizon(file_name_rnn_64_20, mape_per_step_64_20, file_name_rnn_64_20, True, show_plot=True)
