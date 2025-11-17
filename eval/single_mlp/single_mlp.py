import numpy as np
import pandas as pd
from pathlib import Path
from utils.load_data import (
    load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs
)
from models.single_mlp.single_mlp import load_model
import random
from analysis.model_analysis import ModelPlotCreator


# Names of lag feature columns used as input (previous load values).
LAG_COLS = ["load_timestamp_-1", "load_timestamp_-2", "load_timestamp_-3"]


def calc_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE) in percent.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE expressed in percent.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)


def row_to_x(row, feature_cols, lag_buffer_std):
    """
    Build a feature vector for a single timestamp, injecting current lag values.

    The original lag columns in the row are replaced with values from
    the provided lag buffer.

    Parameters
    ----------
    row : pandas.Series
        Row from the ML DataFrame for a single timestamp.
    feature_cols : list[str]
        Ordered list of feature column names used by the model.
    lag_buffer_std : list[float]
        Current lag values [t-1, t-2, t-3] in standardized form.

    Returns
    -------
    np.ndarray
        Feature vector for the model, dtype float32.
    """
    x = row[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for j, col in enumerate(LAG_COLS):
        if col in feature_cols:
            idx = feature_cols.index(col)
            x[idx] = lag_buffer_std[j]
    return x


def recursive_20h_from(df_ml, start_ts, model, feature_cols):
    """
    Run a recursive 20-hour ahead forecast starting from a given timestamp.

    At each step:
    - the model predicts load at time t,
    - the prediction is fed back into the lag buffer for the next step.

    All computations are done in standardized space; values are decoded later.

    Parameters
    ----------
    df_ml : pandas.DataFrame
        ML DataFrame with features and standardized 'load'.
    start_ts : pandas.Timestamp
        Starting timestamp of the forecast window.
    model : tf.keras.Model
        Trained Single MLP model.
    feature_cols : list[str]
        List of feature column names used by the model.

    Returns
    -------
    (np.ndarray, np.ndarray)
        y_true_std : true standardized loads over the 20-hour horizon.
        y_pred_std : predicted standardized loads over the 20-hour horizon.
    """
    horizon = 20
    idxs = pd.date_range(start_ts, periods=horizon, freq="h")

    # Initial lag buffer from the starting timestamp.
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

        # Shift the lag buffer: new t-1 is the current prediction.
        lag_buffer = [yhat_std, lag_buffer[0], lag_buffer[1]]

    return np.array(y_true_std, dtype=np.float32), np.array(y_pred_std, dtype=np.float32)


def evaluate_recursive(
    model_path="models/single_mlp/models/single_mlp_25neurons_20epochs.keras",
    start_hour=1,
    plot=False,
):
    """
    Evaluate the Single MLP in a recursive 20-hour setting across the test set.

    For each valid starting timestamp with the given hour:
    - run a 20-hour recursive forecast,
    - decode standardized values back to MW,
    - compute MAPE over the 20-hour horizon.

    The function reports:
    - global MAPE across all windows,
    - MAPE by weekday,
    - saves results to CSV,
    - optionally plots 3 random forecast examples.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved .keras model, by default
        "models/single_mlp/models/single_mlp_25neurons_20epochs.keras".
    start_hour : int, optional
        Hour of day (0–23) used as the starting hour for forecast windows,
        by default 1.
    plot : bool, optional
        If True, generate plots for 3 random forecast windows, by default False.
    """
    params = DataLoadingParams()
    df_train_ml, df_train_raw = load_training_data(params)
    df_test_ml, df_test_raw = load_test_data(params)

    df_test_ml = df_test_ml.sort_index()
    feature_cols = [c for c in df_test_ml.columns if c != "load"]

    model = load_model(model_path)

    # Collect starting timestamps for which a full 20-hour horizon exists.
    start_points = []
    for ts in df_test_ml.index:
        if ts.hour != start_hour:
            continue
        idxs = pd.date_range(ts, periods=20, freq="h")
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
            "MAPE20": mape20,
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

    # Add global MAPE as an extra row.
    out_df = weekday_mape.to_frame(name="MAPE20")
    out_df.loc["GLOBAL"] = mean_total

    out_df.to_csv(out_path)
    print(f"\n💾 Zapisano: {out_path}")

    # ==========================
    # 3 random forecast examples
    # ==========================

    if not plot:
        return

    if not start_points:
        print("\n⚠️ Brak dostępnych startów do narysowania.")
        return

    n_examples = min(3, len(start_points))
    chosen_starts = random.sample(start_points, n_examples)

    plotter = ModelPlotCreator()

    print(f"\n📈 Rysuję {n_examples} losowe przykłady prognozy (po 20 godzin każde):")
    for i, ts in enumerate(chosen_starts):
        yt_std, yp_std = recursive_20h_from(df_test_ml, ts, model, feature_cols)

        yt = decode_ml_outputs(yt_std.reshape(-1, 1), df_train_raw).reshape(-1)
        yp = decode_ml_outputs(yp_std.reshape(-1, 1), df_train_raw).reshape(-1)

        print(f" → przykład {i+1}: start {ts} ({ts.day_name()})")

        plotter.plot_predictions(
            y_real=yt,
            y_pred=yp,
            model_name=f"{model_name}_example_{i+1}",
            folder="single_mlp",
            freq="h",
            save_plot=True,
            show_plot=False,
        )


if __name__ == "__main__":
    evaluate_recursive(
        model_path="models/single_mlp/models/single_mlp_25neurons_20epochs.keras",
        start_hour=1,
        plot=False,
    )
