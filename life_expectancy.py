import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('life expectancy.csv')
india = data[data['Country'] == 'India']
plt.scatter(india['Year'], india['Life expectancy '], marker='o')
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy of India over Years")
plt.show()
