import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

results_file = os.path.join(
    BASE_DIR, "data", "processed", "evaluation_results.csv"
)

output_file = os.path.join(
    BASE_DIR, "data", "processed", "llm_judge_results.csv"
)

df = pd.read_csv(results_file)

sample = df.head(30).copy()


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


results = []

for i, row in sample.iterrows():

    customer_message = str(row["customer_message"])
    predicted_intent = str(row["predicted_intent"])

    reply = generate_reply(predicted_intent)

    prompt = f"""
You are evaluating an automated Amazon customer support agent.

Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

Drafted support reply:
{reply}

Evaluate the reply using:

1. Correctness: Does it appropriately address the customer's issue?
2. Groundedness: Is it consistent with the predicted intent and
historical Amazon support workflow?
3. Helpfulness: Does it provide a useful next step?

Give each score from 1 to 5.

Return ONLY valid JSON:

{{
    "correctness": 1,
    "groundedness": 1,
    "helpfulness": 1,
    "overall": 1,
    "reason": "short explanation"
}}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a strict customer support quality evaluator."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.choices[0].message.content.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        result = json.loads(text)

        results.append({
            "customer_message": customer_message,
            "predicted_intent": predicted_intent,
            "suggested_reply": reply,
            "correctness": result["correctness"],
            "groundedness": result["groundedness"],
            "helpfulness": result["helpfulness"],
            "overall": result["overall"],
            "reason": result["reason"]
        })

        print(f"Evaluated {len(results)}/{len(sample)}")

    except Exception as e:

        print(f"Error on example {i}: {e}")


output = pd.DataFrame(results)

output.to_csv(output_file, index=False)

print("\n======================================")
print("LLM AS JUDGE")
print("======================================")

if len(output) > 0:

    print(f"Examples evaluated: {len(output)}")
    print(
        f"Average correctness: "
        f"{output['correctness'].mean():.2f}"
    )
    print(
        f"Average groundedness: "
        f"{output['groundedness'].mean():.2f}"
    )
    print(
        f"Average helpfulness: "
        f"{output['helpfulness'].mean():.2f}"
    )
    print(
        f"Average overall: "
        f"{output['overall'].mean():.2f}"
    )

print("\nSaved to:")
print(output_file)