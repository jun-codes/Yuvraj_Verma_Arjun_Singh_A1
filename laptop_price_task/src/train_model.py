import os
import pickle
import numpy as np
import pandas as pd
from data_preprocessing import preprocess, drop_exact_duplicates

raw_data = "laptop_price_task/data/train_data.csv"   
model_dir = "laptop_price_task/models"
model_path = os.path.join(model_dir, "regression_model_4.pkl")

target = "Price"

selected_features = [
    "SSD", "ppi", "Ips",
    "Ram",
    "Hybrid",
    "Weight",
    "Inches", "Touchscreen",
    "gpu_AMD", "gpu_Intel", "gpu_Nvidia",
    "cpu_brand_AMD Processor", "cpu_brand_Intel Core i3",
    "cpu_brand_Intel Core i5", "cpu_brand_Intel Core i7",
    "cpu_brand_Other Intel Processor",
    "company_Acer", "company_Apple", "company_Asus", "company_Dell",
    "company_HP", "company_Lenovo", "company_MSI",
    "company_Others", "company_Toshiba", "company_Samsung",
    "type_Gaming", "type_Netbook", "type_Notebook","type_Ultrabook", "type_Workstation", "type_2 in 1 Convertible",
    ]


def predict(features, weights, bias): #predicted value
    return np.dot(features, weights) + bias

def loss_function(weights, bias, features, target):
    y_pred = predict(features, weights, bias)
    return np.mean((target - y_pred) ** 2)

def gradient_descent_step(weights, bias, x, y, lr, alpha): #Lasso Regularization
    N = x.shape[0]
    y_pred = predict(x, weights, bias)
    errors = y_pred - y

    weight_gradient = (2 / N) * np.dot(x.T, errors)
    weight_gradient += alpha * np.sign(weights)

    bias_gradient = (2 / N) * np.sum(errors)

    return weights - lr * weight_gradient, bias - lr * bias_gradient

raw = pd.read_csv(raw_data)
raw.columns = raw.columns.str.strip()

raw = drop_exact_duplicates(raw)
processed_data = preprocess(raw)
processed_data[target] = pd.to_numeric(processed_data[target], errors="raise")

for col in selected_features:
    if col not in processed_data.columns:
        processed_data[col] = 0

x = processed_data[selected_features].copy()
y_train = processed_data[target].values.astype(float)


x_min = x.min(axis=0)
x_max = x.max(axis=0)
den = (x_max - x_min).replace(0, 1)
x_scaled = (x - x_min) / den

x_train = x_scaled.values.astype(float)


n_features = x_train.shape[1]
w = np.zeros(n_features, dtype=float)
b = 0.0

learning_rate = 0.1      
alpha = 0.01              
num_iterations = 20000

for i in range(num_iterations):
    w, b = gradient_descent_step(w, b, x_train, y_train, learning_rate, alpha)
    if i % 1000 == 0:
        mse = loss_function(w, b, x_train, y_train)
        print(f"iteration {i:5d}  MSE={mse:.2f}")


final_mse = loss_function(w, b, x_train, y_train)
final_rmse = np.sqrt(final_mse)
y_pred = predict(x_train, w, b)

ss_res = float(np.sum((y_train - y_pred)**2))
ss_tot = float(np.sum((y_train - y_train.mean())**2))
r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

print("\nFinal metrics on training data")
print(f"MSE:  {final_mse:.2f}")
print(f"RMSE: {final_rmse:.2f}")
print(f"R^2:  {r2:.4f}")

os.makedirs(model_dir, exist_ok=True)
artifact = {
    "weights": w,
    "bias": b,
    "features": selected_features,
    "scaler_min_": x_min.to_dict(),
    "scaler_max_": x_max.to_dict(),
    "alpha": alpha,
    "learning_rate": learning_rate
}
with open(model_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nSaved model to {model_path}")
