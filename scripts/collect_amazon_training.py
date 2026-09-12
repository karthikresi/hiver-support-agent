import pandas as pd

input_file = "data/raw/twcs/twcs.csv"
output_file = "data/processed/amazonhelp_training.csv"

print("Reading dataset...")

df = pd.read_csv(
    input_file,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    dtype={
        "tweet_id": "string",
        "in_response_to_tweet_id": "string",
        "author_id": "string"
    }
)

print("Dataset loaded!")

# AmazonHelp support replies
support = df[
    (df["author_id"] == "AmazonHelp") &
    (df["in_response_to_tweet_id"].notna())
].copy()

print("AmazonHelp replies:", len(support))

# Create tweet lookup
tweet_lookup = df.set_index("tweet_id")["text"]

# Find customer message
support["customer_text"] = support["in_response_to_tweet_id"].map(tweet_lookup)

# Keep matched conversations
result = support[
    support["customer_text"].notna()
][[
    "in_response_to_tweet_id",
    "customer_text",
    "text",
    "created_at"
]]

# Remove duplicate customer messages
result = result.drop_duplicates(subset="customer_text")

# Take 5000 conversations
result = result.head(5000)

result.to_csv(output_file, index=False)

print("\nTraining dataset created!")
print("Number of conversations:", len(result))
print("Saved to:", output_file)