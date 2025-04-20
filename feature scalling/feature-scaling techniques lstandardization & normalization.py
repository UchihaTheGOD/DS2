import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv('wine.csv', header=None, usecols=[0, 1, 2], skiprows=1)
df.columns = ['classlabel', 'Alcohol', 'Malic Acid']
print("Original DataFrame:")
print(df)
scaling = MinMaxScaler()
scaled_value = scaling.fit_transform(df[['Alcohol', 'Malic Acid']])
df[['Alcohol', 'Malic Acid']] = scaled_value
print("\nDataFrame after Min-Max Scaling:")
print(df)
scaling = StandardScaler()
scaled_standard_value = scaling.fit_transform(df[['Alcohol', 'Malic Acid']])
df[['Alcohol', 'Malic Acid']] = scaled_standard_value
print("\nDataFrame after Standard Scaling:")
print(df)
