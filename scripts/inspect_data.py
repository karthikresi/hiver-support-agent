import pandas as pd

file_path = "data/raw/twcs/twcs.csv"

df = pd.read_csv(file_path, nrows=5)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df)

print("\nShape:")
print(df.shape)