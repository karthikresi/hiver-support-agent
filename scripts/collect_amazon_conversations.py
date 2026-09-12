import pandas as pd

file_path = "data/raw/twcs/twcs.csv"
brand = "AmazonHelp"

N = 500

brand_replies = []

print("Collecting AmazonHelp replies...")

# Find AmazonHelp replies
for chunk in pd.read_csv(file_path, chunksize=100000):

    rows = chunk[
        (chunk["author_id"] == brand) &
        (chunk["in_response_to_tweet_id"].notna())
    ]

    for _, row in rows.iterrows():

        brand_replies.append({
            "customer_tweet_id": int(row["in_response_to_tweet_id"]),
            "amazon_tweet_id": int(row["tweet_id"]),
            "amazon_reply": row["text"]
        })

        if len(brand_replies) >= N:
            break

    if len(brand_replies) >= N:
        break


print("AmazonHelp replies collected:", len(brand_replies))


# Get customer tweet IDs
customer_ids = set(
    x["customer_tweet_id"]
    for x in brand_replies
)


# Find customer messages
customer_messages = {}

print("Finding customer messages...")

for chunk in pd.read_csv(file_path, chunksize=100000):

    matches = chunk[
        chunk["tweet_id"].isin(customer_ids)
    ]

    for _, row in matches.iterrows():

        customer_messages[int(row["tweet_id"])] = row["text"]

    if len(customer_messages) == len(customer_ids):
        break


# Create final dataset
data = []

for item in brand_replies:

    customer_id = item["customer_tweet_id"]

    if customer_id in customer_messages:

        data.append({
            "customer_tweet_id": customer_id,
            "amazon_tweet_id": item["amazon_tweet_id"],
            "customer_message": customer_messages[customer_id],
            "amazon_reply": item["amazon_reply"]
        })


df = pd.DataFrame(data)

output_file = "data/processed/amazonhelp_conversations.csv"

df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Total conversations:", len(df))

print("\nFirst 10 conversations:")
print(df.head(10).to_string(index=False))