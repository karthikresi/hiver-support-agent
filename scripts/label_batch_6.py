import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

df["intent"] = df["intent"].astype("object")

labels = {
    176: "damaged_or_missing_items",
    177: "other",
    178: "other",
    179: "device_issue",
    180: "content_issue",
    181: "device_issue",
    182: "other",
    183: "other",
    184: "delivery_delayed",
    185: "other",
    186: "other",
    187: "other",
    188: "other",
    189: "delivery_preference",
    190: "damaged_or_missing_items",
    191: "other",
    192: "other",
    193: "delivery_delayed",
    194: "prime_billing",
    195: "prime_billing",
    196: "prime_billing",
    197: "other",
    198: "device_issue",
    199: "package_not_received",
    200: "delivery_delayed"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 6 labels saved successfully!")
print("Rows 176-200 labeled.")