import pandas as pd

input_file = "data/processed/amazonhelp_conversations.csv"
output_file = "data/golden/golden_set.csv"

df = pd.read_csv(input_file)

# Take the first 200 examples
golden = df.head(200).copy()

# Add empty intent column
golden["intent"] = ""

# Keep only useful columns
golden = golden[
    [
        "customer_tweet_id",
        "amazon_tweet_id",
        "customer_message",
        "amazon_reply",
        "intent"
    ]
]

golden.to_csv(output_file, index=False)

print("Golden set created!")
print("Examples:", len(golden))
print("File:", output_file)