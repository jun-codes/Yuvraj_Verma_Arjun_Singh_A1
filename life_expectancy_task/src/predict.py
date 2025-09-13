import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data

with open("models/regression_model2.pkl", "rb") as f:
    model2 = pickle.load(f)

weights = model2["weights"]
bias = model2["bias"]
selected_features_list = model2["features"]

raw_data = pd.read_csv("data/train_data.csv")
raw_data.columns = raw_data.columns.str.strip()
data = preprocess_data(raw_data)

features = data[selected_features_list]
features_scaled = (features - features.min()) / (features.max() - features.min())
X_test = features_scaled.values
y_test = data["Life expectancy"].values

def predict(features, weights, bias):
    return np.dot(features, weights) + bias

all_predictions = predict(X_test, weights, bias)

final_mse = np.mean((y_test - all_predictions) ** 2)
final_rmse = np.sqrt(final_mse)

residual_sum = np.sum((y_test - all_predictions) ** 2)
total_sum = np.sum((y_test - np.mean(y_test)) ** 2)
r2_score = 1 - (residual_sum / total_sum)

print(f"MSE: {final_mse:.2f}")
print(f"RMSE: {final_rmse:.2f} years")
print(f"R^2 score: {r2_score:.2f}")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, all_predictions, alpha=0.3, color="pink")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
plt.xlabel("actual life expectancy")
plt.ylabel("predicted life expectancy")
plt.title("actual vs predicted life expectancy")
plt.grid(True)
plt.show()
