import pandas as pd

file_path = "data/raw/twcs/twcs.csv"
brand = "AmazonHelp"

# Step 1: Get 20 AmazonHelp replies
brand_replies = []

for chunk in pd.read_csv(file_path, chunksize=100000):

    rows = chunk[chunk["author_id"] == brand]

    rows = rows[rows["in_response_to_tweet_id"].notna()]

    for _, row in rows.iterrows():

        brand_replies.append({
            "brand_tweet_id": int(row["tweet_id"]),
            "customer_tweet_id": int(row["in_response_to_tweet_id"]),
            "brand_text": row["text"]
        })

        if len(brand_replies) >= 20:
            break

    if len(brand_replies) >= 20:
        break


# Step 2: Collect customer tweet IDs
customer_ids = set()

for reply in brand_replies:
    customer_ids.add(reply["customer_tweet_id"])


# Step 3: Find those customer tweets
customer_messages = {}

for chunk in pd.read_csv(file_path, chunksize=100000):

    matches = chunk[chunk["tweet_id"].isin(customer_ids)]

    for _, row in matches.iterrows():

        customer_messages[int(row["tweet_id"])] = row["text"]

    if len(customer_messages) == len(customer_ids):
        break


# Step 4: Display customer -> AmazonHelp pairs
print("\n========== AMAZONHELP CUSTOMER CONVERSATIONS ==========\n")

for reply in brand_replies:

    customer_id = reply["customer_tweet_id"]

    print("--------------------------------------------")

    print("Customer Tweet ID:", customer_id)
    print("Customer:")
    print(customer_messages.get(customer_id, "Customer message not found"))

    print("\nAmazonHelp:")
    print(reply["brand_text"])