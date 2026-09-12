import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)
df["intent"] = df["intent"].astype("object")

labels = {
    151: "delivery_preference",
    152: "payment_issue",
    153: "content_issue",
    154: "delivery_preference",
    155: "content_issue",
    156: "other",
    157: "payment_issue",
    158: "payment_issue",
    159: "delivery_preference",
    160: "device_issue",
    161: "other",
    162: "order_cancellation",
    163: "other",
    164: "other",
    165: "delivery_preference",
    166: "package_not_received",
    167: "other",
    168: "damaged_or_missing_items",
    169: "payment_issue",
    170: "delivery_delayed",
    171: "delivery_preference",
    172: "other",
    173: "payment_issue",
    174: "other",
    175: "prime_billing",
    176: "other",
    177: "delivery_preference",
    178: "account_access",
    179: "delivery_delayed",
    180: "damaged_or_missing_items",
    181: "other",
    182: "other",
    183: "device_issue",
    184: "prime_video_issue",
    185: "other",
    186: "damaged_or_missing_items",
    187: "package_not_received",
    188: "other",
    189: "payment_issue",
    190: "delivery_delayed",
    191: "payment_issue",
    192: "shipping_information",
    193: "other",
    194: "delivery_delayed",
    195: "other",
    196: "delivery_delayed",
    197: "package_not_received",
    198: "delivery_delayed",
    199: "order_cancellation",
    200: "package_not_received"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 4 labels saved successfully!")
print("Rows 151-200 labeled.")