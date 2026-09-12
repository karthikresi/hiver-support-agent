import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("data/processed/training_clean.csv")
golden = pd.read_csv("data/golden/golden_set.csv")
history = pd.read_csv("data/processed/amazonhelp_training.csv")


# ============================================================
# 2. TRAIN INTENT CLASSIFIER
# ============================================================

X_train = train["customer_text"].fillna("")
y_train = train["intent"]

classifier_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

X_train_vec = classifier_vectorizer.fit_transform(X_train)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X_train_vec, y_train)


# ============================================================
# 3. BUILD RETRIEVAL INDEX
# ============================================================

history_text = history["customer_text"].fillna("")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

history_matrix = retrieval_vectorizer.fit_transform(history_text)


# ============================================================
# 4. INTENT EVALUATION
# ============================================================

X_test = golden["customer_message"].fillna("")
y_test = golden["intent"]

X_test_vec = classifier_vectorizer.transform(X_test)

predictions = classifier.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("\n======================================")
print("INTENT CLASSIFICATION")
print("======================================")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 5. RETRIEVAL EVALUATION
# ============================================================

retrieval_scores = []

for message in X_test:

    query_vector = retrieval_vectorizer.transform([message])

    scores = cosine_similarity(
        query_vector,
        history_matrix
    )[0]

    best_score = scores.max()

    retrieval_scores.append(best_score)


average_similarity = sum(retrieval_scores) / len(retrieval_scores)

good_matches = sum(
    score >= 0.20
    for score in retrieval_scores
)

retrieval_coverage = good_matches / len(retrieval_scores)


print("\n======================================")
print("RETRIEVAL")
print("======================================")

print(
    "Average similarity:",
    round(average_similarity, 4)
)

print(
    "Good historical matches:",
    good_matches,
    "/",
    len(retrieval_scores)
)

print(
    "Retrieval coverage:",
    round(retrieval_coverage, 4)
)


# ============================================================
# 6. ESCALATION POLICY
# ============================================================

risky_intents = {
    "payment_issue",
    "account_access",
    "order_cancellation"
}

escalated = 0
auto_handled = 0

escalation_reasons = []

for i, message in enumerate(X_test):

    intent = predictions[i]

    confidence = classifier.predict_proba(
        classifier_vectorizer.transform([message])
    )[0].max()

    similarity = retrieval_scores[i]


    # High-risk intents
    if intent in risky_intents:

        escalated += 1
        escalation_reasons.append(
            "High-risk intent"
        )


    # Very weak historical match
    elif similarity < 0.20:

        escalated += 1
        escalation_reasons.append(
            "Weak historical match"
        )


    # Both classifier and retrieval are uncertain
    elif confidence < 0.20 and similarity < 0.30:

        escalated += 1
        escalation_reasons.append(
            "Low confidence and weak match"
        )


    # Otherwise automatically handle
    else:

        auto_handled += 1
        escalation_reasons.append(
            "Safe historical match"
        )


escalation_rate = escalated / len(X_test)
auto_handle_rate = auto_handled / len(X_test)


print("\n======================================")
print("ESCALATION")
print("======================================")

print("Escalated:", escalated)
print("Auto-handled:", auto_handled)

print(
    "Escalation rate:",
    round(escalation_rate, 4)
)

print(
    "Auto-handle rate:",
    round(auto_handle_rate, 4)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results = golden.copy()

results["predicted_intent"] = predictions
results["retrieval_similarity"] = retrieval_scores
results["escalation_reason"] = escalation_reasons
results["escalated"] = [
    reason != "Safe historical match"
    for reason in escalation_reasons
]

results.to_csv(
    "data/processed/evaluation_results.csv",
    index=False
)


print("\n======================================")
print("SAVED")
print("======================================")

print(
    "Results saved to:",
    "data/processed/evaluation_results.csv"
)