from eval.cnn.cnn import evaluate_models
from analysis.model_analysis import ModelPlotCreator

# Temporary file to perform model evaluation

file_name = 'CONV_64_DENSE_32-16_168_24_50_32_1h'
y_real, y_pred = evaluate_models(168, 24, file_name, freq="1h")

plot_creator = ModelPlotCreator()
examples = [6, 90, 1000, 4255, 5555, 6789, 7171, 8989, 11111, 14567, 16161]
for i, example in enumerate(examples):
    plot_creator.plot_predictions(y_real[example], y_pred[example],  f'{file_name}-Ex{i+1}', "cnn", freq='1h', save_plot=True, show_plot=True)
