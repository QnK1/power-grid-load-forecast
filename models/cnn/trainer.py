from models.cnn.cnn import get_model, get_model_name, train_model, FEATURE_COLUMNS
from analysis.model_analysis import ModelPlotCreator

convolution_layers = [64]
kernel_sizes = [3]
dense_layers = [32, 16]
sequence_length = 2 * 168
prediction_length = 24

model = get_model(convolution_layers, kernel_sizes, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS))
predicted_label = 'load'
epochs = 50
file_name = get_model_name(convolution_layers, dense_layers, sequence_length, prediction_length, epochs, 32, "1h")

history = train_model(model, sequence_length, prediction_length, predicted_label, epochs, file_name=file_name, freq="1h")

mpc = ModelPlotCreator()
mpc.plot_learning_curves(history, file_name, "cnn", True, True)

