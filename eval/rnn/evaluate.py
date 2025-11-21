from eval.rnn.rnn import model_evaluate
from analysis.model_analysis import ModelPlotCreator

file_name_rnn_64_20 = 'RNN_model_NEURONS-64_EPOCHS-20_SEQUENCE-48_FREQ-1h'

y_pred_64_20, y_true_64_20, mape_per_step_64_20 = model_evaluate(48, 24, file_name=file_name_rnn_64_20)

plot_creator = ModelPlotCreator()

plot_creator.plot_predictions(y_true_64_20[0], y_pred_64_20[0], f"{file_name_rnn_64_20}-Ex1", 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true_64_20[1000], y_pred_64_20[1000], f"{file_name_rnn_64_20}-Ex2", 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true_64_20[500], y_pred_64_20[500], f"{file_name_rnn_64_20}-Ex3", 'rnn', freq='1h', show_plot=True)
plot_creator.plot_predictions(y_true_64_20[200], y_pred_64_20[200], f"{file_name_rnn_64_20}-Ex4", 'rnn', freq='1h', show_plot=True)

plot_creator.plot_mape_over_horizon("RNN_model_NEURONS-64_EPOCHS-20", mape_per_step_64_20,"rnn", True, show_plot=True)