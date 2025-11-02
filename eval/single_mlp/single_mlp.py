import numpy as np
import pandas as pd
from utils.load_data import (
    load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs
)
from models.single_mlp.single_mlp import load_model

LAG_COLS = ["load_timestamp_-1", "load_timestamp_-2", "load_timestamp_-3"]

def calc_mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)

def row_to_x(row, feature_cols, lag_buffer_std):
    """Buduje wektor X z wiersza df_ml, ale z PODMIANĄ lagów na bieżący bufor predykcji (w skali std)."""
    x = row[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for j, col in enumerate(LAG_COLS):
        if col in feature_cols:
            idx = feature_cols.index(col)
            x[idx] = lag_buffer_std[j]
    return x

def recursive_20h_from(df_ml, start_ts, model, feature_cols):
    """
    Rekurencyjnie przewiduje 20 godzin od start_ts:
      - w kroku 0 używa prawdziwych lagów z wiersza startowego,
      - w kolejnych krokach podstawia DO X swoje poprzednie predykcje.
    Zwraca: (y_true_std[20], y_pred_std[20]).
    """
    horizon = 20
    idxs = pd.date_range(start_ts, periods=horizon, freq="h")
    # bufor lagów w skali STANDARYZOWANEJ (takiej jak trenowanie)
    lag_buffer = [
        df_ml.loc[start_ts, "load_timestamp_-1"],
        df_ml.loc[start_ts, "load_timestamp_-2"],
        df_ml.loc[start_ts, "load_timestamp_-3"],
    ]
    y_true_std, y_pred_std = [], []

    for t in idxs:
        row = df_ml.loc[t]
        x = row_to_x(row, feature_cols, lag_buffer)
        yhat_std = model.predict(x.reshape(1, -1), verbose=0).reshape(-1)[0]
        y_pred_std.append(yhat_std)
        # prawdziwy target w skali std z df_ml:
        y_true_std.append(row["load"])
        # aktualizacja bufora lagów (nowa predykcja staje się lagiem -1 w nast. kroku)
        lag_buffer = [yhat_std, lag_buffer[0], lag_buffer[1]]

    return np.array(y_true_std), np.array(y_pred_std)

def evaluate_recursive(
    model_path="models/single_mlp/trained/single_mlp_25neurons_20epochs.keras",
    start_hour=1,
    eval_days=100
):
    print("Ładowanie danych (train scaler + test features)…")
    params = DataLoadingParams()
    df_train_ml, df_train_raw = load_training_data(params)  # scaler do decode
    df_test_ml,  df_test_raw  = load_test_data(params)      # cechy i target (standaryzowane)

    # lista cech wejściowych (wszystko poza 'load')
    feature_cols = [c for c in df_test_ml.columns if c != "load"]

    print("Ładowanie modelu…")
    model = load_model(model_path)

    # wybierz ostatnie eval_days dni w test i startuj o 01:00 każdego dnia
    end_day = df_test_ml.index.max().normalize()
    start_day = end_day - pd.Timedelta(days=eval_days-1)
    start_points = pd.date_range(start_day, end_day, freq="D") \
                     .map(lambda d: d + pd.Timedelta(hours=start_hour))

    # tylko punkty, które fizycznie są w indexie i mają pełne 20h dalej
    start_points = [ts for ts in start_points
                    if ts in df_test_ml.index and ts + pd.Timedelta(hours=19) in df_test_ml.index]

    records = []
    for ts in start_points:
        # rekurencyjna ścieżka 20h w skali std
        yt_std, yp_std = recursive_20h_from(df_test_ml, ts, model, feature_cols)
        # odskaluje do MW – UWAGA: używamy SCALERA Z TRAIN (df_train_raw), tak jak w Twoim decode_ml_outputs
        yt = decode_ml_outputs(yt_std.reshape(-1, 1), df_train_raw).reshape(-1)
        yp = decode_ml_outputs(yp_std.reshape(-1, 1), df_train_raw).reshape(-1)
        dow = ts.day_name()
        mape20 = calc_mape(yt, yp)
        records.append({"start_ts": ts, "weekday": dow, "MAPE20": mape20})

    if not records:
        raise SystemExit("Brak punktów startowych do ewaluacji (sprawdź index i start_hour).")

    df_res = pd.DataFrame(records)
    # MAPE średnie po dniach tygodnia (group_keys=False eliminuje FutureWarning)
    weekday_mape = df_res.groupby("weekday", group_keys=False)["MAPE20"].mean().sort_index()
    mean_total = float(df_res["MAPE20"].mean())

    print(f"\n📊 Globalny MAPE (20h): {mean_total:.2f}%\n")
    print("📅 MAPE wg dnia tygodnia (20h):")
    print(weekday_mape.round(2))

    out_path = "eval/single_mlp/results/results_mlp_20h_25neurons.csv"
    weekday_mape.to_csv(out_path, header=["MAPE20"])
    print(f"\n💾 Zapisano: {out_path}")

if __name__ == "__main__":
    evaluate_recursive(model_path="models/single_mlp/trained/single_mlp_25neurons_20epochs.keras",
                       start_hour=1,
                       eval_days=100
    )
