import pandas as pd

file_path = "data/golden/golden_set.csv"

df = pd.read_csv(file_path)

df["intent"] = df["intent"].astype("object")

labels = {
    76: "package_not_received",
    77: "delivery_delayed",
    78: "payment_issue",
    79: "payment_issue",
    80: "payment_issue",
    81: "payment_issue",
    82: "other",
    83: "other",
    84: "package_not_received",
    85: "package_not_received",
    86: "package_not_received",
    87: "other",
    88: "other",
    89: "other",
    90: "prime_billing",
    91: "shipping_information",
    92: "shipping_information",
    93: "account_access",
    94: "account_access",
    95: "account_access",
    96: "account_access",
    97: "payment_issue",
    98: "payment_issue",
    99: "package_not_received",
    100: "delivery_delayed",
    101: "other",
    102: "prime_video_issue",
    103: "prime_video_issue",
    104: "prime_video_issue",
    105: "prime_video_issue",
    106: "prime_video_issue",
    107: "account_access",
    108: "delivery_delayed",
    109: "delivery_delayed",
    110: "other",
    111: "device_issue",
    112: "order_change",
    113: "account_access",
    114: "account_access",
    115: "account_access",
    116: "other",
    117: "account_access",
    118: "account_access",
    119: "account_access",
    120: "account_access",
    121: "package_not_received",
    122: "delivery_delayed",
    123: "shipping_information",
    124: "package_not_received",
    125: "shipping_information"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 4 labels saved successfully!")
print("Rows 76-125 labeled.")