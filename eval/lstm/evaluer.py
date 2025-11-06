from eval.lstm.lstm import evaluate_models
from eval.result_plotter import plot_predictions

# Temporary file to perform model evaluation

file_name = 'LSTM_64-32_DENSE_32-16_168_24_50_1h'
y_real, y_pred = evaluate_models(168, 24, file_name, freq="1h")

plot_predictions(y_real, y_pred, [3, 800, 2000, 5000, 7564, 10023], file_name, save_plots=False)