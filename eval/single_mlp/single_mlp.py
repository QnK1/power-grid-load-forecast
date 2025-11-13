import numpy as np
import pandas as pd
from pathlib import Path
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
    x = row[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for j, col in enumerate(LAG_COLS):
        if col in feature_cols:
            idx = feature_cols.index(col)
            x[idx] = lag_buffer_std[j]
    return x


def recursive_20h_from(df_ml, start_ts, model, feature_cols):
    horizon = 20
    idxs = pd.date_range(start_ts, periods=horizon, freq="h")

    # startowy bufor lagów
    lag_buffer = [
        df_ml.loc[start_ts, "load_timestamp_-1"],
        df_ml.loc[start_ts, "load_timestamp_-2"],
        df_ml.loc[start_ts, "load_timestamp_-3"],
    ]

    y_true_std = []
    y_pred_std = []

    for t in idxs:
        row = df_ml.loc[t]  
        x = row_to_x(row, feature_cols, lag_buffer)

        yhat_std = model(x[None, :], training=False).numpy().reshape(-1)[0]
        y_pred_std.append(yhat_std)
        y_true_std.append(row["load"])

        lag_buffer = [yhat_std, lag_buffer[0], lag_buffer[1]]

    return np.array(y_true_std, dtype=np.float32), np.array(y_pred_std, dtype=np.float32)


def evaluate_recursive(
    model_path="models/single_mlp/models/single_mlp_25neurons_20epochs.keras",
    start_hour=1,
):
    params = DataLoadingParams()
    df_train_ml, df_train_raw = load_training_data(params)
    df_test_ml, df_test_raw = load_test_data(params)

    df_test_ml = df_test_ml.sort_index()
    feature_cols = [c for c in df_test_ml.columns if c != "load"]

    model = load_model(model_path)

    # wybieramy tylko takie starty, dla których istnieje KAŻDA z 20 godzin
    start_points = []
    for ts in df_test_ml.index:
        if ts.hour != start_hour:
            continue
        idxs = pd.date_range(ts, periods=20, freq="h")
        # sprawdzamy, czy wszystkie są w indexie
        if all(t in df_test_ml.index for t in idxs):
            start_points.append(ts)

    records = []
    for ts in start_points:
        yt_std, yp_std = recursive_20h_from(df_test_ml, ts, model, feature_cols)

        yt = decode_ml_outputs(yt_std.reshape(-1, 1), df_train_raw).reshape(-1)
        yp = decode_ml_outputs(yp_std.reshape(-1, 1), df_train_raw).reshape(-1)

        mape20 = calc_mape(yt, yp)
        records.append({
            "start_ts": ts,
            "weekday": ts.day_name(),
            "MAPE20": mape20
        })

    if not records:
        raise SystemExit("Brak punktów startowych do ewaluacji (DST wywalił wszystko?)")

    df_res = pd.DataFrame(records)
    weekday_mape = df_res.groupby("weekday", group_keys=False)["MAPE20"].mean().sort_index()
    mean_total = float(df_res["MAPE20"].mean())

    print(f"\n📊 Globalny MAPE (całe 20h): {mean_total:.2f}%\n")
    print("📅 MAPE wg dnia tygodnia (całe 20h):")
    print(weekday_mape.round(2))

    out_dir = Path("eval/single_mlp/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem
    out_path = out_dir / f"{model_name}_20h_{start_hour:02d}h_results.csv"

    # tu dokładamy global do tabeli
    out_df = weekday_mape.to_frame(name="MAPE20")
    out_df.loc["GLOBAL"] = mean_total

    out_df.to_csv(out_path)
    print(f"\n💾 Zapisano: {out_path}")



if __name__ == "__main__":
    evaluate_recursive(model_path="models/single_mlp/models/single_mlp_25neurons_15epochs.keras",
                       start_hour=1,
    )
