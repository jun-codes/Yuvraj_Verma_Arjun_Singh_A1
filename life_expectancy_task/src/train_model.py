import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_data

raw_data = pd.read_csv('data/train_data.csv')
raw_data.columns = raw_data.columns.str.strip()
data = preprocess_data(raw_data)

def predict(features, weights, bias):
    return np.dot(features, weights) + bias

def calculate_mse(weights, bias, features, target):
    y_predicted = predict(features, weights, bias)
    return np.mean((target - y_predicted) ** 2)

def gradient_descent(weights_now, bias_now, features, target, learning_rate):
    N = features.shape[0]
    y_predicted = predict(features, weights_now, bias_now)
    errors = y_predicted - target
    weight_gradients = (2/N) * np.dot(features.T, errors)
    bias_gradient = (2/N) * np.sum(errors)
    new_weights = weights_now - learning_rate * weight_gradients
    new_bias = bias_now - learning_rate * bias_gradient
    return new_weights, new_bias

country_features = list(data.filter(like="Country_").columns)

selected_features_list = [
    'Income composition of resources', 'Schooling', 'HIV/AIDS', 'Adult Mortality',
    'BMI', 'Diphtheria', 'Polio', 'thinness  1-19 years', 'GDP', 'Alcohol'
] + country_features

features = data[selected_features_list]
target = data['Life expectancy']

features_scaled = (features - features.min()) / (features.max() - features.min())
X_train = features_scaled.values
y_train = target.values

num_features = X_train.shape[1]
weights = np.zeros(num_features)
bias = 0
learning_rate = 0.25
num_iterations = 50000

for i in range(num_iterations):
    weights, bias = gradient_descent(weights, bias, X_train, y_train, learning_rate)
    if i % 100 == 0:
        current_mse = calculate_mse(weights, bias, X_train, y_train)
        print(f"Iteration {i}: MSE = {current_mse:.4f}")

final_mse = calculate_mse(weights, bias, X_train, y_train)
final_rmse = np.sqrt(final_mse)

all_predictions = predict(X_train, weights, bias)

ss_res = np.sum((y_train - all_predictions) ** 2)
ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f} years")
print(f"R^2 score: {r2_score:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(y_train, all_predictions, alpha=0.3, label='Model Predictions', color='pink')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', linewidth=2, label='Perfect Fit Line')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs. Predicted Life Expectancy")
plt.legend()
plt.grid(True)
plt.show()

print("done")

model2 = {"weights": weights, "bias": bias, "features": selected_features_list}

os.makedirs("life_expectancy_task/models", exist_ok=True)

with open("life_expectancy_task/models/regression_model2.pkl", "wb") as f:
    pickle.dump(model2, f)
