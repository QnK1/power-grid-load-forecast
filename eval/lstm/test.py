from pathlib import Path
from keras.models import load_model

from analysis.model_analysis import ModelPlotCreator

# Load
# file_name = 'LSTM_64_DENSE_32-16_168_24_50_1h'
# Many load
# file_name = 'LSTM_64-32_DENSE_32-16_168_24_51_1h'
# Load
file_name = 'LSTM_64-32_DENSE_32-16_168_24_50_1h'
# project_folder = Path(__file__).parent.parent.parent
# model_path = project_folder / 'models' / 'lstm' / 'models' / f'{file_name}.keras'
# model = load_model(model_path)
#
# print(model.summary())
plot_creator = ModelPlotCreator()
mape_values = [
    2.3930043797514093,
    2.3385799880693434,
    2.442425758399181,
    2.6537235208433825,
    2.7073828510753883,
    2.7580838173193265,
    2.976424939401572,
    2.9562109261193865,
    3.0079862605813252,
    3.0849373006205916,
    3.0774254671619703,
    3.04820380568584,
    3.0668894259041193,
    3.1624857547520278,
    3.2542527229053353,
    3.3947090703662135,
    3.335541367864,
    3.30337287000996,
    3.277910867493566,
    3.2262499369851114,
    3.2273190952218487,
    3.166494045270495,
    3.133787931060275,
    3.1370931444611108
]

# teraz możesz wywołać funkcję
plot_creator.plot_mape_over_horizon(
    model_name=file_name,
    mape_values=mape_values,
    folder="lstm",
    save_plot=True,
    show_plot=True
)