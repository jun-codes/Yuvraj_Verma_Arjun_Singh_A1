# laptop_price_task/src/predict.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess  

with open(r"laptop_price_task\models\regression_model_3.pkl", "rb") as f:
    model = pickle.load(f)

weights = model["weights"]
bias = model["bias"]
selected_features = model["features"]  

raw_data = pd.read_csv(r"laptop_price_task\data\train_data.csv")
raw_data.columns = raw_data.columns.str.strip()
proc_data = preprocess(raw_data)

for col in selected_features:
    if col not in proc_data.columns:
        proc_data[col] = 0

X = proc_data[selected_features].copy()
y_test = proc_data["Price"].astype(float).values

if "scaler_min_" in model and "scaler_max_" in model:
    X_min = pd.Series(model["scaler_min_"])
    X_max = pd.Series(model["scaler_max_"])
    X_min = X_min.reindex(selected_features)
    X_max = X_max.reindex(selected_features)
else:
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

den = (X_max - X_min).replace(0, 1.0)  
X_scaled = (X - X_min) / den
X_scaled = X_scaled.fillna(0.0)        
X_test = X_scaled.values.astype(float)


def predict(features, weights, bias):
    return np.dot(features, weights) + bias

y_pred = predict(X_test, weights, bias)

mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 score: {r2:.2f}")


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
