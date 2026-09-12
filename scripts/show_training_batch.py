import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)

for i, row in df.iloc[:50].iterrows():
    print(f"\n{i + 1}. {row['customer_text']}")