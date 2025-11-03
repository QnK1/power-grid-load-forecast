from eval.lstm.lstm import evaluate_models
from eval.result_plotter import plot_predictions

# Temporary file to perform model evaluation

file_name = 'LSTM_64-32_DENSE_16-16_168_24_50'
y_real, y_pred = evaluate_models(168, 24, file_name)

plot_predictions(y_real, y_pred, [10, 43, 235, 3425], file_name)