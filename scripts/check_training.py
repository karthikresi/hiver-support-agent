import pandas as pd

file_path = "data/processed/amazonhelp_training.csv"

df = pd.read_csv(file_path)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 customer messages:")
for i, text in enumerate(df["customer_text"].head(10), 1):
    print(f"{i}. {text}")

print("\nMissing values:")
print(df.isna().sum())