import pandas as pd
import os

INPUT_FILE = "data/processed/evaluation_results.csv"
OUTPUT_FILE = "data/processed/human_review.csv"

df = pd.read_csv(INPUT_FILE)

# Select 15 examples
sample = df.sample(
    n=15,
    random_state=42
).copy()

# Generate the same type of reply used by the agent
def generate_reply(intent):

    replies = {
        "delivery_delayed":
            "Sorry about the delay. Please check your order tracking and the carrier status. If the package is still delayed, please contact Amazon support for further help.",

        "package_not_received":
            "Sorry that you have not received your package. Please check the tracking information and confirm the carrier status. If it still cannot be located, please contact Amazon support.",

        "damaged_or_missing_items":
            "Sorry about the issue with your order. Please check the order details and report the missing or damaged item through Amazon support so the issue can be resolved.",

        "shipping_information":
            "Please check your order tracking and shipping details for the latest delivery information. If you need further help, please contact Amazon support.",

        "order_cancellation":
            "Please check whether the order is still eligible for cancellation. If you need assistance, please contact Amazon support.",

        "order_change":
            "Please check your order details to see whether the requested change is still possible. If not, Amazon support can help with available options.",

        "delivery_preference":
            "Please check the available delivery options for your order and select the preferred option if it is available.",

        "prime_billing":
            "Please check your Prime membership and billing details. If there is an unexpected charge, please contact Amazon support for assistance.",

        "account_access":
            "Please use the Amazon account recovery and sign-in support options to restore access to your account. If the issue continues, contact Amazon support.",

        "payment_issue":
            "Please check your payment method and the payment status for the order. If the payment issue continues, please contact Amazon support.",

        "prime_video_issue":
            "Please check your Prime Video playback and device settings. If the problem continues, please contact Amazon support for further troubleshooting.",

        "device_issue":
            "Please check the device settings and follow the available troubleshooting steps. If the issue continues, please contact Amazon support.",

        "content_issue":
            "Please provide more details about the content issue so that the appropriate Amazon support team can assist you.",

        "other":
            "Sorry you are experiencing this issue. Please provide a few more details so Amazon support can determine the best way to help."
    }

    return replies.get(
        intent,
        "Please provide more details so Amazon support can help."
    )


sample["suggested_reply"] = sample["predicted_intent"].apply(
    generate_reply
)

# Add human review columns
sample["human_correct"] = ""
sample["human_grounded"] = ""
sample["human_helpful"] = ""
sample["human_overall"] = ""

# Keep only useful columns
sample = sample[
    [
        "customer_message",
        "intent",
        "predicted_intent",
        "suggested_reply",
        "human_correct",
        "human_grounded",
        "human_helpful",
        "human_overall"
    ]
]

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("======================================")
print("HUMAN REVIEW SET")
print("======================================")

print(f"Examples selected: {len(sample)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nOpen this file in Excel:")
print(OUTPUT_FILE)

print("\nScore each reply:")
print("1 = Poor")
print("2 = Weak")
print("3 = Acceptable")
print("4 = Good")
print("5 = Excellent")