import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

df["intent"] = df["intent"].astype("object")

labels = {
    126: "other",
    127: "shipping_information",
    128: "shipping_information",
    129: "delivery_preference",
    130: "delivery_delayed",
    131: "delivery_delayed",
    132: "damaged_or_missing_items",
    133: "damaged_or_missing_items",
    134: "other",
    135: "other",
    136: "damaged_or_missing_items",
    137: "order_cancellation",
    138: "package_not_received",
    139: "shipping_information",
    140: "shipping_information",
    141: "package_not_received",
    142: "delivery_delayed",
    143: "other",
    144: "other",
    145: "payment_issue",
    146: "payment_issue",
    147: "content_issue",
    148: "other",
    149: "other",
    150: "other",
    151: "other",
    152: "package_not_received",
    153: "other",
    154: "delivery_preference",
    155: "package_not_received",
    156: "account_access",
    157: "payment_issue",
    158: "other",
    159: "other",
    160: "other",
    161: "other",
    162: "device_issue",
    163: "other",
    164: "other",
    165: "account_access",
    166: "payment_issue",
    167: "other",
    168: "package_not_received",
    169: "content_issue",
    170: "delivery_delayed",
    171: "delivery_delayed",
    172: "delivery_delayed",
    173: "delivery_delayed",
    174: "content_issue",
    175: "other"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 5 labels saved successfully!")
print("Rows 126-175 labeled.")