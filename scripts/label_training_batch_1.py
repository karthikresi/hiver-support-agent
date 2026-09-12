import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)

df["intent"] = df["intent"].astype("object")

labels = {
    1: "damaged_or_missing_items",
    2: "delivery_delayed",
    3: "other",
    4: "other",
    5: "delivery_delayed",
    6: "delivery_delayed",
    7: "other",
    8: "device_issue",
    9: "package_not_received",
    10: "other",
    11: "account_access",
    12: "other",
    13: "payment_issue",
    14: "other",
    15: "payment_issue",
    16: "other",
    17: "delivery_delayed",
    18: "other",
    19: "delivery_delayed",
    20: "other",
    21: "other",
    22: "delivery_delayed",
    23: "device_issue",
    24: "package_not_received",
    25: "other",
    26: "prime_video_issue",
    27: "other",
    28: "other",
    29: "damaged_or_missing_items",
    30: "other",
    31: "other",
    32: "other",
    33: "delivery_delayed",
    34: "other",
    35: "other",
    36: "shipping_information",
    37: "other",
    38: "damaged_or_missing_items",
    39: "delivery_delayed",
    40: "other",
    41: "other",
    42: "delivery_delayed",
    43: "package_not_received",
    44: "package_not_received",
    45: "shipping_information",
    46: "other",
    47: "other",
    48: "shipping_information",
    49: "other",
    50: "other"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 1 labels saved successfully!")
print("Rows 1-50 labeled.")