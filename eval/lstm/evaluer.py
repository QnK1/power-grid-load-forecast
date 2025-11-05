from eval.lstm.lstm import evaluate_models
from eval.result_plotter import plot_predictions

# Temporary file to perform model evaluation

file_name = 'LSTM_8_DENSE_8_84_12_20'
y_real, y_pred = evaluate_models(84, 12, file_name, freq="2h")

plot_predictions(y_real, y_pred, [3, 800, 2000, 5000, 7564, 10023, 29012, 35555, 43244, 51051, 58085], file_name, save_plots=False)