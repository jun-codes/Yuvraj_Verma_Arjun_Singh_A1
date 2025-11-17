import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_data

raw_data = pd.read_csv('data/train_data.csv')
raw_data.columns = raw_data.columns.str.strip()
data = preprocess_data(raw_data) #don't commit data   

def loss_function(weights, bias, features, target):
    y_predicted = predict(features, weights, bias)
    return np.mean((target - y_predicted) ** 2)

def gradient_descent(weights_now, bias_now, features, target, learning_rate):
    N = features.shape[0]
    y_predicted = predict(features, weights_now, bias_now)
    errors = y_predicted - target
    weight_gradients = (2/N) * np.dot(features.T, errors) #vectorized using numpy = quicker
    bias_gradient = (2/N) * np.sum(errors)
    new_weights = weights_now - learning_rate * weight_gradients
    new_bias = bias_now - learning_rate * bias_gradient
    return new_weights, new_bias

def predict(features, weights, bias):
    return np.dot(features, weights) + bias

status_features = list(data.filter(like="Status_").columns)
country_features = list(data.filter(like="Country_").columns)

selected_features_list = [
    'Income composition of resources', 'Schooling', 'HIV/AIDS', 'Adult Mortality',
    'BMI', 'Diphtheria', 'Polio', 'thinness  1-19 years', 'GDP', 'Alcohol'
] + country_features + status_features #satus/country 1 hot encoded

for col in selected_features_list:
    if col not in data.columns:
        data[col] = 0  

data = data[selected_features_list]
features = data[selected_features_list]
target = data['Life expectancy']

features_scaled = (features - features.min()) / (features.max() - features.min())
X_train = features_scaled.values
y_train = target.values

num_features = X_train.shape[1]
weights = np.zeros(num_features)
bias = 0
learning_rate = 0.15
num_iterations = 50000 # didn't require this many plateus around 30k

for i in range(num_iterations):
    weights, bias = gradient_descent(weights, bias, X_train, y_train, learning_rate)
    if i % 100 == 0:
        current_mse = loss_function(weights, bias, X_train, y_train)
        print(f"Iteration {i}: MSE = {current_mse:.2f}")

final_mse = loss_function(weights, bias, X_train, y_train)
final_rmse = np.sqrt(final_mse)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.3, color="pink", label="Predictions")
mn = float(min(y_test.min(), y_pred.min()))
mx = float(max(y_test.max(), y_pred.max()))
plt.plot([mn, mx], [mn, mx], "--r", linewidth=2, label="Perfect fit")
plt.xlabel("Actual life Expectancy")
plt.ylabel("Predicted life Expectancy")
plt.title("Actual vs Predicted life Expectancy")
plt.legend()
plt.grid(True) #graph with true fit line
plt.show()
print("done")

model3 = {"weights": weights, "bias": bias, "features": selected_features_list}

os.makedirs("life_expectancy_task/models", exist_ok=True)

with open("life_expectancy_task/models/regression_model3.pkl", "wb") as f:
    pickle.dump(model3, f) #byte steam pickle file
