import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)
df["intent"] = df["intent"].astype("object")

labels = {
    101: "other",
    102: "shipping_information",
    103: "other",
    104: "other",
    105: "delivery_delayed",
    106: "delivery_delayed",
    107: "other",
    108: "prime_billing",
    109: "other",
    110: "other",
    111: "delivery_delayed",
    112: "other",
    113: "package_not_received",
    114: "delivery_preference",
    115: "other",
    116: "other",
    117: "other",
    118: "other",
    119: "other",
    120: "other",
    121: "other",
    122: "damaged_or_missing_items",
    123: "delivery_delayed",
    124: "delivery_delayed",
    125: "payment_issue",
    126: "device_issue",
    127: "other",
    128: "delivery_delayed",
    129: "payment_issue",
    130: "prime_video_issue",
    131: "damaged_or_missing_items",
    132: "other",
    133: "damaged_or_missing_items",
    134: "other",
    135: "payment_issue",
    136: "other",
    137: "account_access",
    138: "other",
    139: "other",
    140: "delivery_delayed",
    141: "damaged_or_missing_items",
    142: "delivery_delayed",
    143: "other",
    144: "package_not_received",
    145: "other",
    146: "damaged_or_missing_items",
    147: "package_not_received",
    148: "other",
    149: "shipping_information",
    150: "payment_issue"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 3 labels saved successfully!")
print("Rows 101-150 labeled.")
