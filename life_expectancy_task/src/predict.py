import argparse
import pickle
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

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
    selected_features_list = model["features"]

    raw_data = pd.read_csv(args.data_path)
    raw_data.columns = raw_data.columns.str.strip()
    data = preprocess_data(raw_data)

    features = data[selected_features_list]
    features_scaled = (features - features.min()) / (features.max() - features.min())
    X_test = features_scaled.values
    y_test = data["Life expectancy"].values

    all_predictions = predict(X_test, weights, bias)

    final_mse = np.mean((y_test - all_predictions) ** 2)
    final_rmse = np.sqrt(final_mse)
    residual_sum = np.sum((y_test - all_predictions) ** 2)
    total_sum = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (residual_sum / total_sum)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, all_predictions, alpha=0.3, color="pink")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
    plt.xlabel("actual life expectancy")
    plt.ylabel("predicted life expectancy")
    plt.title("actual vs predicted life expectancy")
    plt.grid(True)
    plt.show()

    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {final_mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {final_rmse:.2f}\n")
        f.write(f"R-squared (R^2) Score: {r2_score:.2f}\n")

    pd.Series(all_predictions).to_csv(
        args.predictions_output_path, index=False, header=False
    )

    print("results saved.")

if __name__ == "__main__":
    main()
