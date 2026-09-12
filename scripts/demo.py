import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("data/processed/training_clean.csv")
history = pd.read_csv("data/processed/amazonhelp_training.csv")

X_train = train["customer_text"].fillna("")
y_train = train["intent"]


# ============================================================
# 2. TRAIN INTENT CLASSIFIER
# ============================================================

classifier_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

X = classifier_vectorizer.fit_transform(X_train)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X, y_train)


# ============================================================
# 3. BUILD HISTORICAL RETRIEVAL INDEX
# ============================================================

history_text = history["customer_text"].fillna("")
history_reply = history["text"].fillna("")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

history_matrix = retrieval_vectorizer.fit_transform(history_text)


# ============================================================
# 4. RETRIEVE SIMILAR HISTORICAL CASES
# ============================================================

def retrieve(query, top_k=3):

    query_vector = retrieval_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        history_matrix
    )[0]

    indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in indices:
        results.append({
            "customer": history_text.iloc[index],
            "reply": history_reply.iloc[index],
            "score": scores[index]
        })

    return results


# ============================================================
# 5. GENERATE GROUNDED REPLY
# ============================================================

def generate_reply(intent):

    if intent in {
        "delivery_delayed",
        "package_not_received"
    }:
        return (
            "I'm sorry your package hasn't arrived as expected. "
            "Please check the current tracking status and carrier information. "
            "If the tracking does not resolve the issue, please contact "
            "Amazon support so they can look into the order."
        )

    if intent == "payment_issue":
        return (
            "I'm sorry you're having trouble with your payment. "
            "Please check your payment details and current payment status. "
            "If the issue continues, please contact Amazon support for assistance."
        )

    if intent == "account_access":
        return (
            "I'm sorry you're having trouble accessing your account. "
            "Please use Amazon's account support flow to check the issue "
            "and regain access."
        )

    if intent == "prime_video_issue":
        return (
            "I'm sorry you're having trouble with Prime Video. "
            "Please check the playback issue and the device you're using. "
            "If the problem continues, please contact Amazon support."
        )

    if intent == "device_issue":
        return (
            "I'm sorry you're having trouble with your device. "
            "Please follow the relevant device troubleshooting steps. "
            "If the issue continues, please contact Amazon support."
        )

    if intent == "delivery_preference":
        return (
            "Thanks for contacting Amazon. "
            "Please check your delivery options for the order. "
            "If you need further assistance, please contact Amazon support."
        )

    if intent == "shipping_information":
        return (
            "Thanks for contacting Amazon. "
            "Please check the order tracking information for the latest "
            "shipping and delivery details."
        )

    if intent == "order_cancellation":
        return (
            "Please check your order details to see whether cancellation "
            "is still available. If you need further assistance, please "
            "contact Amazon support."
        )

    return (
        "Thanks for contacting Amazon. "
        "Please provide more details about the issue so we can help."
    )


# ============================================================
# 6. ESCALATION DECISION
# ============================================================

def decide_escalation(intent, confidence, similarity):

    risky_intents = {
        "payment_issue",
        "account_access",
        "order_cancellation"
    }

    if intent in risky_intents:
        return True, "Sensitive account, payment, or order action"

    if confidence < 0.50:
        return True, "Low intent confidence"

    if similarity < 0.20:
        return True, "Low similarity to historical support cases"

    return False, "High confidence and relevant historical match"


# ============================================================
# 7. COMPLETE SUPPORT AGENT
# ============================================================

def run_agent(message):

    # ----- Intent classification -----

    vector = classifier_vectorizer.transform([message])

    probabilities = classifier.predict_proba(vector)[0]

    best_index = probabilities.argmax()

    intent = classifier.classes_[best_index]
    confidence = probabilities[best_index]


    # ----- Historical retrieval -----

    results = retrieve(message, top_k=3)

    best_match = results[0]

    similarity = best_match["score"]


    # ----- Generate response -----

    reply = generate_reply(intent)


    # ----- Escalation -----

    escalate, reason = decide_escalation(
        intent,
        confidence,
        similarity
    )


    return {
        "intent": intent,
        "confidence": confidence,
        "historical_match": best_match["customer"],
        "historical_reply": best_match["reply"],
        "similarity": similarity,
        "suggested_reply": reply,
        "escalate": escalate,
        "reason": reason
    }


# ============================================================
# 8. RUN DEMO
# ============================================================

if __name__ == "__main__":

    message = input("\nCustomer message: ")

    result = run_agent(message)

    print("\n==============================")
    print("SUPPORT AGENT RESULT")
    print("==============================")

    print("\nIntent:")
    print(result["intent"])

    print("\nIntent confidence:")
    print(round(result["confidence"], 3))

    print("\nHistorical customer match:")
    print(result["historical_match"])

    print("\nHistorical AmazonHelp response:")
    print(result["historical_reply"])

    print("\nHistorical similarity:")
    print(round(result["similarity"], 3))

    print("\nSuggested grounded reply:")
    print(result["suggested_reply"])

    print("\nEscalate:")
    print(result["escalate"])

    print("\nEscalation reason:")
    print(result["reason"])