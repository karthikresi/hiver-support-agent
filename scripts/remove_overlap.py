import pandas as pd

train_file = "data/processed/training_sample.csv"
golden_file = "data/golden/golden_set.csv"
output_file = "data/processed/training_clean.csv"

train = pd.read_csv(train_file)
golden = pd.read_csv(golden_file)

golden_text = set(
    golden["customer_message"].fillna("").astype(str).str.strip()
)

train["clean_text"] = (
    train["customer_text"].fillna("").astype(str).str.strip()
)

clean = train[~train["clean_text"].isin(golden_text)].copy()

clean = clean.drop(columns=["clean_text"])

clean.to_csv(output_file, index=False)

print("Original training examples:", len(train))
print("Removed overlapping examples:", len(train) - len(clean))
print("Clean training examples:", len(clean))
print("Saved to:", output_file)