from eval.rnn.rnn import model_evaluate
from analysis.model_analysis import ModelPlotCreator
from utils.load_data import load_test_data, load_training_data, DataLoadingParams, decode_ml_outputs

y_pred, y_true = model_evaluate(168, 24, file_name='rnn_32_10')
print(y_pred)
print(y_true)

plot_creator = ModelPlotCreator()

plot_creator.plot_predictions(y_true[15], y_pred[15], 'rnn_32_10', 'rnn', freq='1h', show_plot=True)