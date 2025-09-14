import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess


def predict(features, weights, bias):
    return np.dot(features, weights) + bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metrics_output_path", required=True)
    parser.add_argument("--predictions_output_path", required=True)
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    weights = model["weights"]
    bias = model["bias"]
    # remove duplicate feature names (keep first occurrence)
    selected_features = list(dict.fromkeys(model["features"]))

    raw_data = pd.read_csv(args.data_path)
    raw_data.columns = raw_data.columns.str.strip()
    proc_data = preprocess(raw_data)

    # add missing features as zeros
    for col in selected_features:
        if col not in proc_data.columns:
            proc_data[col] = 0

    # drop duplicate columns in case preprocess introduced them
    X = proc_data[selected_features].copy()
    X = X.loc[:, ~X.columns.duplicated()]

    y_test = proc_data["Price"].astype(float).values

    if "scaler_min_" in model and "scaler_max_" in model:
        X_min = pd.Series(model["scaler_min_"]).reindex(X.columns)
        X_max = pd.Series(model["scaler_max_"]).reindex(X.columns)
    else:
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

    den = (X_max - X_min).replace(0, 1.0)
    X_scaled = (X - X_min) / den
    X_scaled = X_scaled.fillna(0.0)
    X_test = X_scaled.values.astype(float)

    y_pred = predict(X_test, weights, bias)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R^2 score: {r2:.2f}\n")

    pd.Series(y_pred).to_csv(
        args.predictions_output_path, index=False, header=False
    )

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color="pink", label="Predictions")
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "--r", linewidth=2, label="Perfect fit")
    plt.xlabel("Actual Laptop Price")
    plt.ylabel("Predicted Laptop Price")
    plt.title("Actual vs Predicted Laptop Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("results saved.")


if __name__ == "__main__":
    main()
