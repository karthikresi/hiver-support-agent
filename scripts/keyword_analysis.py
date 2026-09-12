import pandas as pd

file_path = "data/processed/amazonhelp_conversations.csv"

df = pd.read_csv(file_path)

keywords = {
    "delivery": ["delivery", "delivered", "arrive", "arrived", "late"],
    "package": ["package", "parcel", "shipping"],
    "order": ["order", "preorder"],
    "prime": ["prime"],
    "payment": ["payment", "charge", "charged", "refund"],
    "account": ["account", "login", "log in"],
    "video": ["video", "movie", "film", "playback"],
    "device": ["echo", "alexa", "fire tv"],
    "cancel": ["cancel", "cancellation"],
}

text = df["customer_message"].fillna("").str.lower()

print("Total conversations:", len(df))

print("\nKeyword occurrence:")

for category, words in keywords.items():

    count = 0

    for word in words:
        count += text.str.contains(word, regex=False).sum()

    print(category, "->", count)