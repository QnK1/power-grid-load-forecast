from eval.lstm.lstm import evaluate_models
from analysis.model_analysis import ModelPlotCreator

# Temporary file to perform model evaluation

file_name = 'LSTM_16-8_DENSE__168_24_50_1h'
y_real, y_pred = evaluate_models(168, 24, file_name, freq="1h")

plot_creator = ModelPlotCreator()
examples = [6, 90, 1000]
for i, example in enumerate(examples):
    plot_creator.plot_predictions(y_real[example], y_pred[example],  f'{file_name}-Ex{i+1}', "lstm", freq='1h', save_plot=False, show_plot=True)
