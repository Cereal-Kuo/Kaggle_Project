import pandas as pd
import numpy as py

def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (col,
              df[col].dtype,
              df[col].isnull().sum(),
              df[col].count(),
              df[col].nunique(),
              df[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows) 
    df.columns = ['feature','dtype','nan','count','nunique','unique']
    return df
