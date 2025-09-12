import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(data):
    print("Missing values per column:")
    print(data.isnull().sum())

    columns_with_missing_data = [
        'Life expectancy', 'Adult Mortality', 'Alcohol', 'Hepatitis B', 'BMI',
        'Polio', 'Total expenditure', 'Diphtheria', 'GDP', 'Population',
        'thinness  1-19 years', 'thinness 5-9 years',
        'Income composition of resources', 'Schooling'
    ]

    for col in columns_with_missing_data:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in '{col}'.")

    print(data.isnull().sum())
    print("preprocessing complete")
    return data