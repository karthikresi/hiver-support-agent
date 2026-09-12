import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

# Make sure intent can store text
df["intent"] = df["intent"].astype("object")

labels = [
    "other",
    "delivery_delayed",
    "other",
    "order_change",
    "package_not_received",
    "other",
    "other",
    "other",
    "delivery_delayed",
    "prime_billing",
    "package_not_received",
    "delivery_delayed",
    "order_cancellation",
    "prime_billing",
    "prime_billing",
    "account_access",
    "account_access",
    "prime_billing",
    "payment_issue",
    "account_access",
    "other",
    "delivery_delayed",
    "other",
    "other",
    "content_issue"
]

df.loc[25:49, "intent"] = labels

df.to_csv(file_path, index=False)

print("Batch 2 labelled successfully.")
print("\nRows 26-50:")
print(df.loc[25:49, ["customer_message", "intent"]].to_string(index=False))