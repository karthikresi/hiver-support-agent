import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

# Make sure intent can store text
df["intent"] = df["intent"].astype("object")

labels = [
    "content_issue",
    "content_issue",
    "other",
    "other",
    "device_issue",
    "device_issue",
    "device_issue",
    "device_issue",
    "package_not_received",
    "package_not_received",
    "order_change",
    "device_issue",
    "other",
    "device_issue",
    "device_issue",
    "shipping_information",
    "package_not_received",
    "shipping_information",
    "delivery_delayed",
    "other",
    "other",
    "delivery_delayed",
    "delivery_delayed",
    "delivery_delayed",
    "payment_issue"
]

df.loc[50:74, "intent"] = labels

df.to_csv(file_path, index=False)

print("Batch 3 labelled successfully.")
print("\nRows 51-75:")
print(df.loc[50:74, ["customer_message", "intent"]].to_string(index=False))