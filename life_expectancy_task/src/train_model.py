import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_data

raw_data = pd.read_csv('data/train_data.csv')
raw_data.columns = raw_data.columns.str.strip()
data = preprocess_data(raw_data)

def predict(features_row, weights, bias):
    prediction = bias
    for i in range(len(weights)):
        prediction += weights[i] * features_row.iloc[i]
    return prediction

def calculate_mse(weights, bias, features, target):
    total_error = 0
    for i in range(len(features)):
        y_actual = target.iloc[i]
        y_predicted = predict(features.iloc[i], weights, bias)
        total_error += (y_actual - y_predicted)**2
    return total_error / float(len(features))

def gradient_descent(weights_now, bias_now, features, target, learning_rate):
    weight_gradients = [0] * len(weights_now)
    bias_gradient = 0
    N = float(len(features))

    for i in range(len(features)):
        y_actual = target.iloc[i]
        y_predicted = predict(features.iloc[i], weights_now, bias_now)
        error = y_predicted - y_actual
        
        bias_gradient += (2/N) * error
        for j in range(len(weights_now)):
            weight_gradients[j] += (2/N) * error * features.iloc[i].iloc[j]
    
    new_weights = [weights_now[j] - learning_rate * weight_gradients[j] for j in range(len(weights_now))]
    new_bias = bias_now - learning_rate * bias_gradient
    
    return new_weights, new_bias

country_features = list(data.filter(like="Country_").columns)

selected_features_list = [
    'Income composition of resources', 'Schooling', 'HIV/AIDS', 'Adult Mortality',
    'BMI', 'Diphtheria', 'Polio', 'thinness  1-19 years', 'GDP', 'Alcohol'
]

features = data[selected_features_list]
target = data['Life expectancy']

features_scaled = (features - features.min()) / (features.max() - features.min())
X_train = features_scaled
y_train = target

num_features = X_train.shape[1] #no of columns
weights = [0] * num_features
bias = 0
learning_rate = 0.2
num_iterations = 400

for i in range(num_iterations):
    weights, bias = gradient_descent(weights, bias, X_train, y_train, learning_rate)
    if i % 100 == 0:
        current_mse = calculate_mse(weights, bias, X_train, y_train)
        print(f"Iteration {i}: MSE = {current_mse:.4f}")

final_mse = calculate_mse(weights, bias, X_train, y_train)
final_rmse = np.sqrt(final_mse)



all_predictions = [predict(X_train.iloc[i], weights, bias) for i in range(len(X_train))]
ss_res = np.sum((y_train - all_predictions) ** 2)
ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"Final MSE: {final_mse:.4f}")
print(f"Final RMSE: {final_rmse:.4f} years")
print(f"R^2 Score: {r2_score:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(y_train, all_predictions, alpha=0.3, label='Model Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', linewidth=2, label='Perfect Fit Line')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs. Predicted Life Expectancy")
plt.legend()
plt.grid(True)
plt.show()

print("done")


model1 = {"weights": weights, "bias": bias, "features": selected_features_list}
import os

os.makedirs("life_expectancy_task/models", exist_ok=True)

with open("life_expectancy_task/models/regression_model1.pkl", "wb") as f:
    pickle.dump(model1, f)
