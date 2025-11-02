from models.lstm.lstm import get_model, get_model_name, train_model, FEATURE_COLUMNS

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


lstm_layers = [64, 32]
dense_layers = [16, 16]
sequence_length = 168
prediction_length = 24
model = get_model(lstm_layers, dense_layers, sequence_length, prediction_length, len(FEATURE_COLUMNS), dropout=0.3, recurrent_dropout=0.2)
predicted_label = 'load'
epochs = 50
file_name = get_model_name(lstm_layers, dense_layers, sequence_length, prediction_length, epochs)

train_model(model, sequence_length, prediction_length, predicted_label, epochs, file_name=file_name)