import pandas as pd

file_path = "data/raw/twcs/twcs.csv"

brand_counts = {}

for chunk in pd.read_csv(file_path, chunksize=100000):

    counts = chunk["author_id"].value_counts()

    for author, count in counts.items():
        brand_counts[author] = brand_counts.get(author, 0) + count

print("Total unique authors:", len(brand_counts))

print("\nTop 20 authors/accounts:")

top_authors = sorted(
    brand_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]

for author, count in top_authors:
    print(author, "->", count)