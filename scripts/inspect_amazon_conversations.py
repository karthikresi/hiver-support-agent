import pandas as pd

file_path = "data/processed/amazonhelp_conversations.csv"

df = pd.read_csv(file_path)

print("Total conversations:", len(df))

print("\nCustomer messages:")
for i, text in enumerate(df["customer_message"].head(50), 1):

    print("\n", i, "->", text)

print("\nAmazonHelp replies:")
for i, text in enumerate(df["amazon_reply"].head(20), 1):

    print("\n", i, "->", text)