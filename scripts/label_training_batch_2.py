import pandas as pd

file_path = "data/processed/training_sample.csv"

df = pd.read_csv(file_path)
df["intent"] = df["intent"].astype("object")

labels = {
    51: "delivery_delayed",
    52: "prime_billing",
    53: "shipping_information",
    54: "other",
    55: "prime_billing",
    56: "other",
    57: "package_not_received",
    58: "delivery_delayed",
    59: "shipping_information",
    60: "other",
    61: "delivery_preference",
    62: "delivery_preference",
    63: "payment_issue",
    64: "other",
    65: "other",
    66: "delivery_delayed",
    67: "other",
    68: "account_access",
    69: "other",
    70: "other",
    71: "other",
    72: "delivery_preference",
    73: "other",
    74: "other",
    75: "shipping_information",
    76: "other",
    77: "other",
    78: "delivery_delayed",
    79: "order_cancellation",
    80: "other",
    81: "other",
    82: "other",
    83: "other",
    84: "device_issue",
    85: "other",
    86: "other",
    87: "delivery_delayed",
    88: "package_not_received",
    89: "other",
    90: "delivery_delayed",
    91: "payment_issue",
    92: "other",
    93: "other",
    94: "other",
    95: "other",
    96: "delivery_preference",
    97: "payment_issue",
    98: "delivery_delayed",
    99: "delivery_preference",
    100: "payment_issue"
}

for row, label in labels.items():
    df.loc[row - 1, "intent"] = label

df.to_csv(file_path, index=False)

print("Batch 2 labels saved successfully!")
print("Rows 51-100 labeled.")