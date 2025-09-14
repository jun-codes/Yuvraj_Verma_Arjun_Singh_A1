import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()


    numerical_features = [
        'age', 
        'membership_years', 
        'number_of_children', 
        'product_rating',
        'product_review_count',
        'days_since_last_purchase'
    ]

    categorical_features = [
        'gender', 
        'income_bracket', 
        'loyalty_program', 
        'education_level',
        'occupation',
        'purchase_frequency'
    ]
    
    target_variable = ['avg_purchase_value']

    all_selected_columns = numerical_features + categorical_features + target_variable
    df_selected = df[all_selected_columns]

    # Fill numerical Nans with the mean of the column
    for col in numerical_features:
        df_selected[col].fillna(df_selected[col].mean(), inplace=True)
        
    # Fill categorical Nans with the mode
    for col in categorical_features:
        df_selected[col].fillna(df_selected[col].mode()[0], inplace=True)

    df_processed = pd.get_dummies(df_selected, columns=categorical_features, drop_first=True)

    print("Data preprocessing complete.")
    print(f"Original number of columns: {len(df.columns)}")
    print(f"Columns after processing: {len(df_processed.columns)}")
    
    return df_processed

if __name__ == '__main__':
        raw_data = pd.read_csv('data/Retail.csv') 
        
        processed_data = preprocess_retail_data(raw_data)
        
        print("First 5 rows of processed data:")
        print(processed_data.head())
        
        print("\nData types of processed data:")
        print(processed_data.info())
    

    