import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load historical conversations
df = pd.read_csv("data/processed/amazonhelp_training.csv")

customer_text = df["customer_text"].fillna("")
amazon_reply = df["text"].fillna("")


# Create TF-IDF index
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

matrix = vectorizer.fit_transform(customer_text)


def retrieve(query, top_k=3):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, matrix)[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "customer_message": customer_text.iloc[index],
            "amazon_reply": amazon_reply.iloc[index],
            "score": scores[index]
        })

    return results


# Test
if __name__ == "__main__":

    query = input("Enter customer message: ")

    results = retrieve(query)

    print("\nTop historical matches:\n")

    for i, result in enumerate(results, 1):

        print(f"--- Match {i} ---")
        print("Similarity:", round(result["score"], 3))
        print("Customer:", result["customer_message"])
        print("AmazonHelp:", result["amazon_reply"])
        print()