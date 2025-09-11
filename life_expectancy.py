import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('life expectancy.csv')
data.columns = data.columns.str.strip()

def calculate_mse(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Year_scaled
        y = points.iloc[i]['Life expectancy']
        total_error += (y - (m * x + b))** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i].Year_scaled
        y = points.iloc[i]['Life expectancy']
        m_gradient += -(2/N) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/N) * (y - (m_now * x + b_now))
    m_new = m_now - learning_rate * m_gradient
    b_new = b_now - learning_rate * b_gradient
    return m_new, b_new

training_data = data[['Year', 'Life expectancy']].copy()
training_data.dropna(subset=['Life expectancy'], inplace=True)

min_year = training_data['Year'].min()
training_data['Year_scaled'] = (training_data['Year'] - min_year) / 10.0

m = 0
b = 0
learning_rate = 0.1  
num_iterations = 1000

for i in range(num_iterations):
    m, b = gradient_descent(m, b, training_data, learning_rate)
    if i % 100 == 0:
        current_mse = calculate_mse(m, b, training_data)
        print(f"Iteration {i}: MSE = {current_mse:.4f}")

final_mse = calculate_mse(m, b, training_data)
final_rmse = np.sqrt(final_mse)

print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f} years")

plt.figure(figsize=(12, 8))
plt.scatter(data['Year'], data['Life expectancy'], alpha=0.1, label='(All Countries)')

predicted_y = [m * ((x - min_year) / 10.0) + b for x in data['Year']]
plt.plot(data['Year'], predicted_y, color='red', linewidth=3, label='Linear Fit')

plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Global Life Expectancy vs. Year")
plt.legend()
plt.grid(True)
plt.show()

print("done")