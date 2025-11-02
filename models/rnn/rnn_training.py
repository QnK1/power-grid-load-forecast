from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from models.rnn.rnn import build_model, train_model, model_predict
from utils.load_data import load_training_data, DataLoadingParams

FEATURE_COLUMNS = [
    'load',
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',

]
predicted_label = 'load'

sequence_length = 24
prediction_length = 1
feature_count = len(FEATURE_COLUMNS)
epochs = 40

#Build, train and evaluate model
model = build_model(64, sequence_length, feature_count)
train_model(model, sequence_length, prediction_length, epochs, predicted_label)


#Predict and Plot results for visualization

y_pred,y_true, mape = model_predict(model, sequence_length)

params = DataLoadingParams()
data, raw_data = load_training_data(params)
data = data.iloc[0:800]
index = data.index[sequence_length+prediction_length:]

plt.plot( index, y_pred, color='blue')
plt.plot(index, y_true, color='red')
plt.title(f'RNN Prediction - MAPE {mape*100}%')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.show()