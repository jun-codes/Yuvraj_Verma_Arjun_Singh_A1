import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_data

raw_data = pd.read_csv('data/Retail.csv')  # load dataset
raw_data.columns = raw_data.columns.str.strip()  # clean column names
data = preprocess_data(raw_data)  # custom preprocessing

def loss_function(weights, bias, features, target):
    y_predicted = predict(features, weights, bias)
    return np.mean((target - y_predicted) ** 2)  # mse error

def gradient_descent(weights_now, bias_now, features, target, learning_rate):
    N = features.shape[0]
    y_predicted = predict(features, weights_now, bias_now)
    errors = y_predicted - target  # difference between prediction and truth
    weight_gradients = (2/N) * np.dot(features.T, errors)  # slope for weights
    bias_gradient = (2/N) * np.sum(errors)  # slope for bias
    new_weights = weights_now - learning_rate * weight_gradients  # move weights opposite slope
    new_bias = bias_now - learning_rate * bias_gradient
    return new_weights, new_bias

def predict(features, weights, bias):
    return np.dot(features, weights) + bias  # linear regression rule

numerical_features = ['age','membership_years','number_of_children','product_rating',
    'product_review_count','days_since_last_purchase']

categorical_dummy_features = [col for col in data.columns if 
    col.startswith(('gender_', 'income_bracket_', 'loyalty_program_', 
                   'education_level_', 'occupation_', 'purchase_frequency_'))]

selected_features_list = numerical_features + categorical_dummy_features

features = data[selected_features_list]  # pick only needed columns
target = data['avg_purchase_value']  # target variable

numerical_data = features[numerical_features]
scaling_min = numerical_data.min()
scaling_range = numerical_data.max() - numerical_data.min()
numerical_scaled = (numerical_data - scaling_min) / scaling_range  # min-max scaling

categorical_data = features[categorical_dummy_features]  # dummies already good

features_scaled = pd.concat([numerical_scaled, categorical_data], axis=1)  # join num + cat

X_train = features_scaled.values
y_train = target.values

weights = np.zeros(X_train.shape[1])  # start weights at 0
bias = 0
learning_rate = 0.15
num_iterations = 1000

for i in range(num_iterations):
    weights, bias = gradient_descent(weights, bias, X_train, y_train, learning_rate)
    if i % 100 == 0:
        current_mse = loss_function(weights, bias, X_train, y_train)
        print(f"Iteration {i}: MSE = {current_mse:.2f}")  #watch training progress

final_mse = loss_function(weights, bias, X_train, y_train)
final_rmse = np.sqrt(final_mse)
y_predicted = predict(X_train, weights, bias)

residual_sum = np.sum((y_train - y_predicted) ** 2)
total_sum = np.sum((y_train - np.mean(y_train))** 2)
r2_score = 1 - (residual_sum / total_sum)  # how much variance model explains

print("Regression Metrics:\n")
print(f"Mean Squared Error (MSE): {final_mse:.2f}\n")
print(f"Root Mean Squared Error (RMSE): {final_rmse:.2f}\n")
print(f"R-squared (R^2) Score: {r2_score:.2f}\n")

print("done")

model3 = {"weights": weights, "bias": bias, "features": selected_features_list,
    "scaling_min": scaling_min, "scaling_range": scaling_range}  # store model + scaling info

os.makedirs("retail_task/models", exist_ok=True)
with open("retail_task/models/regression_model2.pkl", "wb") as f:
    pickle.dump(model3, f)  #save
