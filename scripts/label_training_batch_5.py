import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)
df["intent"] = df["intent"].astype("object")

labels = {
    201: "other",
    202: "delivery_delayed",
    203: "account_access",
    204: "other",
    205: "other",
    206: "delivery_delayed",
    207: "other",
    208: "delivery_delayed",
    209: "delivery_delayed",
    210: "other",
    211: "other",
    212: "account_access",
    213: "payment_issue",
    214: "damaged_or_missing_items",
    215: "payment_issue",
    216: "payment_issue",
    217: "damaged_or_missing_items",
    218: "delivery_delayed",
    219: "delivery_delayed",
    220: "delivery_delayed",
    221: "other",
    222: "other",
    223: "shipping_information",
    224: "other",
    225: "payment_issue",
    226: "other",
    227: "payment_issue",
    228: "damaged_or_missing_items",
    229: "shipping_information",
    230: "shipping_information",
    231: "payment_issue",
    232: "account_access",
    233: "other",
    234: "other",
    235: "other",
    236: "other",
    237: "damaged_or_missing_items",
    238: "device_issue",
    239: "damaged_or_missing_items",
    240: "other",
    241: "delivery_delayed",
    242: "other",
    243: "other",
    244: "shipping_information",
    245: "delivery_preference",
    246: "other",
    247: "other",
    248: "delivery_delayed",
    249: "payment_issue",
    250: "package_not_received"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 5 labels saved successfully!")
print("Rows 201-250 labeled.")