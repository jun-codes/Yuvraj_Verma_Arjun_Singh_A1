# src/data_preprocessing.py
import re
import numpy as np
import pandas as pd
import argparse

def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates().reset_index(drop=True)

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    #Ram and Weight
    df['Ram'] = (
        df['Ram']
          .astype(str)
          .str.strip()
          .str.replace('gb', '', case=False, regex=False)
          .str.replace(r'[^0-9]', '', regex=True)
          .replace('', np.nan)
          .astype(float)
          .astype('Int64')   
    )

    df['Weight'] = (
        df['Weight']
          .astype(str)
          .str.strip()
          .str.lower()
          .str.replace('kg', '', regex=False)
          .str.replace(',', '.', regex=False)  
          .str.replace(r'[^0-9\.]', '', regex=True)
          .replace('', np.nan)
          .astype(float)
    )

    # Touchscreen and IPS
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)

    # Screen resolution into x_res and y_res
    sr_nospace = df['ScreenResolution'].astype(str).str.replace(' ', '', regex=False)
    new = sr_nospace.str.split('x', n=1, expand=True)
    df['X_res'] = new[0]
    df['Y_res'] = new[1]
    df['X_res'] = df['X_res'].str.replace(',', '', regex=False)\
                             .str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype(float).astype(int)
    df['Y_res'] = df['Y_res'].str.replace(',', '', regex=False)\
                             .str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype(float).astype(int)

    # PPI
    df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype(float)

    # CPU name/brand
    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(str(x).split()[0:3]))
    def fetch_processor(text):
        if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
            return text
        return 'Other Intel Processor' if str(text).split()[0] == 'Intel' else 'AMD Processor'
    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

    # GPU brand
    df['Gpu brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])
    df = df[df['Gpu brand'] != 'ARM'].reset_index(drop=True)

        # Memory parsing
    df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
    df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
    df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)

    new = df['Memory'].str.split('+', n=1, expand=True)
    df['first']  = new[0].str.strip()
    df['second'] = new[1]
    df['second'].fillna('0', inplace=True)

    df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
    df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
    df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
    df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

    df['first']  = df['first'].str.replace(r'\D', '', regex=True)
    df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in str(x) else 0)
    df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in str(x) else 0)
    df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in str(x) else 0)
    df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in str(x) else 0)
    df['second'] = df['second'].str.replace(r'\D', '', regex=True)

    df['first']  = df['first'].astype(int)
    df['second'] = df['second'].astype(int)

    df['HDD']    = df['first']*df['Layer1HDD']    + df['second']*df['Layer2HDD']
    df['SSD']    = df['first']*df['Layer1SSD']    + df['second']*df['Layer2SSD']
    df['Hybrid'] = df['first']*df['Layer1Hybrid'] + df['second']*df['Layer2Hybrid']
    df['Flash_Storage'] = df['first']*df['Layer1Flash_Storage'] + df['second']*df['Layer2Flash_Storage']

    df.drop(columns=[
        'first','second',
        'Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage',
        'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'
    ], inplace=True)


    def cat_os(inp):
        if inp in ('Windows 10','Windows 7','Windows 10 S'):
            return 'Windows'
        elif inp in ('macOS','Mac OS X'):
            return 'Mac'
        return 'Others/No OS/Linux'
    df['os'] = df['OpSys'].apply(cat_os)

    #One-Hot Encoding
    df = pd.concat([df, pd.get_dummies(df['Gpu brand'],  prefix='gpu').astype(int)], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Cpu brand'],  prefix='cpu_brand').astype(int)], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Company'],   prefix='company').astype(int)], axis=1)
    df = pd.concat([df, pd.get_dummies(df['TypeName'],   prefix='type').astype(int)], axis=1)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    clean = drop_exact_duplicates(raw)
    processed = preprocess(clean)
    processed.to_csv(args.output, index=False)
    print(f"Saved preprocessed file to {args.output}")
