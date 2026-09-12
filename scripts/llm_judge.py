import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, ".env"))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "evaluation_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "llm_judge_results.csv"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Judge first 30 examples
df = df.head(30).copy()


# ============================================================
# REPLY GENERATOR
# ============================================================

def generate_reply(intent):

    replies = {

        "delivery_delayed":
            "Sorry for the delay. Please check the latest tracking status and carrier information. If the package is still delayed, please contact Amazon support so they can check the shipment.",

        "package_not_received":
            "Sorry that you haven't received your package. Please check the latest tracking information and confirm the delivery status. If it shows delivered but you cannot find it, please contact Amazon support.",

        "damaged_or_missing_items":
            "Sorry about the issue with your order. Please check the order details and contact Amazon support so they can help with the damaged or missing item.",

        "shipping_information":
            "Please check your order tracking and shipping details for the latest delivery information. If you need further help, Amazon support can check the shipment.",

        "order_cancellation":
            "Please check whether cancellation is still available for your order. If you need help cancelling it, please contact Amazon support.",

        "order_change":
            "Please check your order details to see whether the requested change is still available. Amazon support can help if the change cannot be made online.",

        "delivery_preference":
            "Please check the available delivery options for your order. If you need help changing the delivery preference, Amazon support can assist.",

        "prime_billing":
            "Please check your Prime membership and billing details. If there is an unexpected charge, Amazon support can review the billing information.",

        "account_access":
            "Please use the Amazon account recovery and sign-in support options to restore access to your account. If the issue continues, contact Amazon support.",

        "payment_issue":
            "Please check your payment method and the payment status for the order. If the payment issue continues, contact Amazon support so they can review it.",

        "prime_video_issue":
            "Please check your Prime Video playback and device settings. If the problem continues, Amazon support can help troubleshoot the issue.",

        "device_issue":
            "Please check the device settings and try the basic troubleshooting steps. If the problem continues, Amazon support can help troubleshoot the device.",

        "content_issue":
            "Please provide more details about the content issue so Amazon support can identify the problem and help you.",

        "other":
            "Please provide a few more details about the issue so Amazon support can understand the problem and help you."
    }

    return replies.get(
        intent,
        "Please provide more details about the issue so Amazon support can help you."
    )


# ============================================================
# LLM JUDGE
# ============================================================

results = []

print("=" * 50)
print("GEMINI LLM AS JUDGE")
print("=" * 50)

for i, row in df.iterrows():

    customer_message = str(row["customer_message"])
    predicted_intent = str(row["predicted_intent"])
    historical_reply = str(row["amazon_reply"])

    suggested_reply = generate_reply(predicted_intent)

    prompt = f"""
You are evaluating an AI customer-support agent.

The agent receives a customer message and drafts a reply.
The historical AmazonHelp reply is provided as evidence of how the brand
handled the issue historically.

Evaluate the proposed reply.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL AMAZONHELP REPLY:
{historical_reply}

PREDICTED INTENT:
{predicted_intent}

PROPOSED AI REPLY:
{suggested_reply}

Score the proposed reply from 1 to 5 on:

1. correctness
Does it appropriately address the customer's issue?

2. groundedness
Is it consistent with the historical support response?

3. helpfulness
Would the customer know what to do next?

4. overall
Overall quality of the response.

Return ONLY valid JSON:

{{
  "correctness": <1-5>,
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "overall": <1-5>,
  "reason": "short explanation"
}}
"""

    try:

        response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        # Remove markdown JSON fences if Gemini adds them
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        judge = json.loads(text)

        results.append({
            "customer_message": customer_message,
            "predicted_intent": predicted_intent,
            "suggested_reply": suggested_reply,
            "correctness": judge["correctness"],
            "groundedness": judge["groundedness"],
            "helpfulness": judge["helpfulness"],
            "overall": judge["overall"],
            "reason": judge.get("reason", "")
        })

        print(f"Example {len(results)}/30 completed")

        time.sleep(1)

    except Exception as e:

        print(f"Error on example {i}: {e}")


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)

print("Examples judged:", len(result_df))

if len(result_df) > 0:

    print(
        "Average correctness:",
        round(result_df["correctness"].mean(), 2)
    )

    print(
        "Average groundedness:",
        round(result_df["groundedness"].mean(), 2)
    )

    print(
        "Average helpfulness:",
        round(result_df["helpfulness"].mean(), 2)
    )

    print(
        "Average overall:",
        round(result_df["overall"].mean(), 2)
    )

print()
print("Saved to:")
print(OUTPUT_FILE)