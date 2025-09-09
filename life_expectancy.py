import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('life expectancy.csv')
india = data[data['Country'] == 'India']
plt.scatter(india['Year'], india['Life expectancy '], marker='o')
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy of India over Years")
plt.show()

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Year
        y = points.iloc[i]['Life expectancy ']
        total_error += (y - (m * x + b))** 2
    return total_error / float(len(points))

india = data[data['Country'] == 'India'].copy()

india['Year_scaled'] = (india['Year'] - 2000) /10

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i].Year_scaled
        y = points.iloc[i]['Life expectancy ']
        m_gradient += -(2/N) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/N) * (y - (m_now * x + b_now))
    m_new = m_now - learning_rate * m_gradient
    b_new = b_now - learning_rate * b_gradient
    return m_new, b_new

m = 0
b = 0
learning_rate = 0.01  
num_iterations = 5000

for i in range(num_iterations):
    m, b = gradient_descent(m, b, india, learning_rate)
    if i % 500 == 0:
        print(f"Iteration {i}: slope m={m:.4f}, intercept b={b:.4f}")


plt.scatter(india['Year'], india['Life expectancy '], marker='o', label='Actual Data')

predicted_y = [m * ((x - 2000) / 10) + b for x in india['Year']]
plt.plot(india['Year'], predicted_y, color='red', label='Linear Fit')

plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Linear Fit for India Life Expectancy")
plt.legend()
plt.show()

print("done")