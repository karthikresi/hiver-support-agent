import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


train = pd.read_csv("data/processed/training_clean.csv")
golden = pd.read_csv("data/golden/golden_set.csv")

X_train = train["customer_text"].fillna("")
y_train = train["intent"]

X_test = golden["customer_message"].fillna("")
y_test = golden["intent"]


# Word features
word_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

# Character features
char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=20000
)

features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])

X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_features, y_train)

pred = model.predict(X_test_features)


print("\n==============================")
print("IMPROVED CLASSIFIER")
print("==============================")

print("Accuracy:", accuracy_score(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro", zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))