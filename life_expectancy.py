import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('life expectancy.csv')
plt.scatter(data['Year'], data['Life expectancy '])
plt.show()
