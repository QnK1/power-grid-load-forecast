from eval.lstm.lstm import evaluate_models
from eval.result_plotter import plot_predictions

# Temporary file to perform model evaluation

file_name = 'LSTM_64_DENSE_32-16_84_12_20'
y_real, y_pred = evaluate_models(84, 12, file_name, freq="2h")

plot_predictions(y_real, y_pred, [3, 10, 42, 43, 235], file_name, save_plots=False)