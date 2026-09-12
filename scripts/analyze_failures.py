import pandas as pd
import os

INPUT_FILE = "data/processed/evaluation_results.csv"
OUTPUT_FILE = "data/processed/failure_analysis.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only incorrect intent predictions
errors = df[df["intent"] != df["predicted_intent"]].copy()

print("\n======================================")
print("FAILURE ANALYSIS")
print("======================================")

print(f"Total evaluation examples: {len(df)}")
print(f"Incorrect predictions: {len(errors)}")
print(f"Correct predictions: {len(df) - len(errors)}")

# 1. Most common true intents that were misclassified
true_errors = (
    errors["intent"]
    .value_counts()
    .reset_index()
)

true_errors.columns = ["true_intent", "error_count"]

# 2. Most common wrong predictions
pred_errors = (
    errors["predicted_intent"]
    .value_counts()
    .reset_index()
)

pred_errors.columns = ["predicted_intent", "error_count"]

# 3. Most common confusion pairs
confusions = (
    errors.groupby(["intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n--------------------------------------")
print("TOP TRUE INTENTS WITH ERRORS")
print("--------------------------------------")

print(true_errors.head(10).to_string(index=False))

print("\n--------------------------------------")
print("TOP WRONG PREDICTIONS")
print("--------------------------------------")

print(pred_errors.head(10).to_string(index=False))

print("\n--------------------------------------")
print("TOP CONFUSION PAIRS")
print("--------------------------------------")

print(confusions.head(10).to_string(index=False))

# Save confusion analysis
confusions.to_csv(OUTPUT_FILE, index=False)

print("\n--------------------------------------")
print("SAVED")
print("--------------------------------------")

print(f"Failure analysis saved to: {OUTPUT_FILE}")