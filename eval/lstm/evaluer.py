from eval.lstm.lstm import evaluate_models
from analysis.model_analysis import ModelPlotCreator

# Temporary file to perform model evaluation

file_name = 'LSTM_64-32_DENSE_32-16_168_24_50_1h'
y_real, y_pred = evaluate_models(168, 24, file_name, freq="1h")

plot_creator = ModelPlotCreator()
plot_creator.plot_predictions(y_real[6], y_pred[6],  file_name, "lstm", freq='1h', save_plots=True)
