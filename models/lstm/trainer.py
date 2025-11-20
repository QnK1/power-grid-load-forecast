from models.lstm.lstm import get_model, get_model_name, train_model, FEATURE_COLUMNS
from analysis.model_analysis import ModelPlotCreator

# Temporary file for creating LSTM models

# lstm_layers = [8]
# dense_layers = [8]
# sequence_length = 24
# prediction_length = 3
# model = get_model(lstm_layers, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0.2, recurrent_dropout=0.2)
# predicted_label = 'load'
# epochs = 10
# file_name = get_model_name(lstm_layers, dense_layers, sequence_length, prediction_length, epochs)
#
# train_model(model, sequence_length, prediction_length, predicted_label, epochs, file_name=file_name)


# lstm_layers = [64, 32]
# dense_layers = [16, 16]
# sequence_length = 168
# prediction_length = 24
# model = get_model(lstm_layers, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0.3, recurrent_dropout=0.2)
# predicted_label = 'load'
# epochs = 50
# file_name = get_model_name(lstm_layers, dense_layers, sequence_length, prediction_length, epochs)

lstm_layers = [16, 8]
dense_layers = []
sequence_length = 168
prediction_length = 24
model = get_model(lstm_layers, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0.2, recurrent_dropout=0.1)
predicted_label = 'load'
epochs = 50
file_name = get_model_name(lstm_layers, dense_layers, sequence_length, prediction_length, epochs, "1h")

history = train_model(model, sequence_length, prediction_length, predicted_label, epochs, file_name=file_name, freq="1h")

mpc = ModelPlotCreator()
mpc.plot_learning_curves(history, file_name, "lstm", True, True)

# lstm_layers = [128, 64]
# dense_layers = [64, 32, 16]
# sequence_length = 4 * 168
# prediction_length = 4 * 24
# model = get_model(lstm_layers, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0.2, recurrent_dropout=0.1)
# predicted_label = 'load'
# epochs = 75
# file_name = get_model_name(lstm_layers, dense_layers, sequence_length, prediction_length, epochs, "15min")
#
# history = train_model(model, sequence_length, prediction_length, predicted_label, epochs, file_name=file_name, freq="15min")
#
# mpc = ModelPlotCreator()
# mpc.plot_learning_curves(history, file_name, "lstm", True, True)
