import pandas as pd

train = pd.read_csv("data/processed/training_sample.csv")
golden = pd.read_csv("data/golden/golden_set.csv")

train_text = set(
    train["customer_text"].fillna("").astype(str).str.strip()
)

golden_text = set(
    golden["customer_message"].fillna("").astype(str).str.strip()
)

overlap = train_text & golden_text

print("Training examples:", len(train))
print("Golden examples:", len(golden))
print("Exact text overlap:", len(overlap))

if overlap:
    print("\nOverlap found.")
    print("These examples will be removed from training.")
else:
    print("\nNo overlap. Safe to continue.")