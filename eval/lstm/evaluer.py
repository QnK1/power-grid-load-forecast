from eval.lstm.lstm import evaluate_models
from analysis.model_analysis import ModelPlotCreator

# Temporary file to perform model evaluation

file_name = 'LSTM_8_DENSE_8_84_12_11_2h'
y_real, y_pred = evaluate_models(84, 12, file_name, freq="2h")

plot_creator = ModelPlotCreator()
plot_creator.plot_predictions(y_real[6], y_pred[6],  file_name, "lstm", freq='2h', save_plot=False, show_plot=True)
