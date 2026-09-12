import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

# Make intent column capable of storing text
df["intent"] = df["intent"].astype("object")

labels = [
    "device_issue",
    "device_issue",
    "other",
    "prime_video_issue",
    "other",
    "package_not_received",
    "account_access",
    "content_issue",
    "other",
    "shipping_information",
    "shipping_information",
    "damaged_or_missing_items",
    "delivery_delayed",
    "other",
    "other",
    "device_issue",
    "device_issue",
    "package_not_received",
    "package_not_received",
    "other",
    "delivery_preference",
    "delivery_preference",
    "damaged_or_missing_items",
    "other",
    "damaged_or_missing_items"
]

df.loc[:24, "intent"] = labels

df.to_csv(file_path, index=False)

print("Batch 1 labelled successfully.")
print("\nLabels:")
print(df.loc[:24, ["customer_message", "intent"]].to_string(index=False))