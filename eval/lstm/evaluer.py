from eval.lstm.lstm import evaluate_models
from eval.result_plotter import plot_predictions

# Temporary file to perform model evaluation

file_name = 'LSTM_64-32_DENSE_16-16_168_24_50'
y_real, y_pred = evaluate_models(168, 24, file_name)

plot_predictions(y_real, y_pred, [3, 10, 42, 43, 235, 888, 2094, 2137, 3000, 3425, 6780, 10000, 15151], file_name)