from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from models.rnn.rnn import build_model, train_model, FEATURE_COLUMNS
from utils.load_data import load_training_data, DataLoadingParams

predicted_label = 'load'
sequence_length = 48
feature_count = len(FEATURE_COLUMNS)
epochs = 15
units = 64

#Build, train and evaluate model
model = build_model(units, sequence_length,24, feature_count)
train_model(model, sequence_length, 24,epochs, units, label=predicted_label)