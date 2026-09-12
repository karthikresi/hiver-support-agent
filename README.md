# Hiver SDE Intern Take-Home Assignment

## Amazon Customer Support Agent

### 1. Problem Framing

The goal is to build a lightweight customer-support agent for Amazon that can:

1. Classify an incoming customer message into a small set of support intents.

2. Find historically similar AmazonHelp conversations.

3. Draft a reply grounded in how similar customer issues were handled historically.

4. Decide whether the message can be auto-handled or should be escalated.

The system is intentionally lightweight and reproducible. It uses classical machine learning and TF-IDF retrieval rather than a large trained neural network.

---

## 2. Dataset

The project uses the Kaggle **Customer Support on Twitter** dataset.

Dataset:

`thoughtvector/customer-support-on-twitter`

The dataset contains customer-support conversations from brands on Twitter/X.

Relevant columns:

- `tweet_id`

- `author_id`

- `inbound`

- `created_at`

- `text`

- `response_tweet_id`

- `in_response_to_tweet_id`

I selected **AmazonHelp** as the brand.

Conversation relationships were reconstructed using:

`in_response_to_tweet_id`

This connects an AmazonHelp reply to the corresponding customer message.

The original dataset is approximately 3 million tweets.

The dataset is licensed under **CC BY-NC-SA 4.0**. The raw dataset is not included in this repository.

The repository includes two small derived processed subsets required for reproducibility:

data/processed/training_clean.csv — 475 manually labelled training examples

data/processed/amazonhelp_training.csv — 5,000 historical AmazonHelp customer/reply pairs

The original full dataset is not redistributed.

---

## 3. Intent Taxonomy

I defined 14 practical customer-support intents:

1. `delivery_delayed`

2. `package_not_received`

3. `damaged_or_missing_items`

4. `shipping_information`

5. `order_cancellation`

6. `order_change`

7. `delivery_preference`

8. `prime_billing`

9. `account_access`

10. `payment_issue`

11. `prime_video_issue`

12. `device_issue`

13. `content_issue`

14. `other`

The taxonomy was deliberately kept small so that the classifier could be implemented and evaluated quickly.

---

# 4. System Architecture

The system follows this pipeline:

Customer message

        |

        v

Intent Classifier

        |

        +--------------------+

        |                    |

        v                    v

Predicted Intent       Historical Retrieval

        |                    |

        +---------+----------+

                  |

                  v

          Reply Generation

                  |

                  v

          Escalation Decision

                  |

          +-------+-------+

          |               |

          v               v

     Auto-handle       Escalate

### Components

### Intent Classification

TF-IDF word n-grams are used to convert customer messages into numerical features.

A Logistic Regression classifier predicts the intent.

### Historical Retrieval

TF-IDF cosine similarity is used to retrieve the most similar historical AmazonHelp customer conversations.

The retrieved examples contain:

- Historical customer message

- Historical AmazonHelp response

### Reply Generation

The final reply is generated using a small rule-based template based on the predicted intent.

Historical retrieved conversations provide grounding/context rather than copying an old response verbatim.

### Escalation

The system escalates when:

- The intent is high-risk, such as payment, account access, or cancellation.

- Intent confidence is very low.

- The historical similarity is too low.

---

# 5. Training and Golden Evaluation Set

A manually labelled training sample of 500 AmazonHelp customer messages was created.

After removing 25 exact customer-message overlaps with the golden evaluation set:

- Initial training examples: 500

- Removed overlaps: 25

- Final training examples: 475

A separate golden evaluation set contains:

**200 manually labelled examples**

The golden set contains the same 14 intents.

The evaluation set was kept separate from the cleaned training examples.

---

# 6. Baselines

Two baselines were evaluated.

## Baseline 1: Majority Class

The simplest baseline predicts the most common class (`other`) for every example.

Results:

- Accuracy: **28.5%**

- Macro F1: **3.17%**

This provides a lower-bound reference for the classifier.

---

## Baseline 2: TF-IDF + Logistic Regression

The main classifier uses:

- TF-IDF features

- Word bigrams

- Logistic Regression

- Balanced class weights

Results on the 200-example golden set:

- Accuracy: **36.0%**

- Macro F1: **23.94%**

This improves over the majority-class baseline.

The model was intentionally kept simple because the goal was to build a reproducible support-agent prototype rather than a production-scale classifier.

---

# 7. Retrieval Results

Historical AmazonHelp conversations are indexed using TF-IDF.

For every customer message, the system retrieves the top 3 historical conversations.

Evaluation results:

- Average similarity: **0.965**

- Retrieval coverage using the selected similarity threshold: **96.5%**

- Good historical matches: **193 / 200**

However, this number should **not** be interpreted as retrieval accuracy.

The retrieval corpus and golden evaluation examples originate from the same underlying AmazonHelp dataset. Therefore, the retrieval score is optimistic and is not an independent benchmark.

This is an important limitation of the current experiment.

---

# 8. Escalation Results

The escalation policy was evaluated on the 200-example golden set.

Results:

- Auto-handled: **162 / 200**

- Escalated: **38 / 200**

- Auto-handle rate: **81%**

- Escalation rate: **19%**

High-risk intents such as payment, account access, and cancellation are intentionally more conservative.

Low confidence and weak historical similarity can also trigger escalation.

---

# 9. Human Evaluation

A separate human review was performed on 15 randomly selected examples.

Each response was scored from 1 to 5 for:

- Correctness

- Groundedness

- Helpfulness

- Overall quality

Results:

| Metric | Average Score |

|---|---:|

| Correctness | 3.07 / 5 |

| Groundedness | 3.73 / 5 |

| Helpfulness | 3.60 / 5 |

| Overall | 3.93 / 5 |

The human review suggests that the generated replies are generally useful and reasonably grounded, but classification errors still affect correctness.

---

# 10. LLM-as-Judge

An LLM-as-judge evaluation was implemented using Gemini.

The judge evaluates:

Correctness

Groundedness

Helpfulness

Overall response quality

The judge successfully evaluated 12 examples before the Gemini free-tier
request quota was reached.

Results:

Metric

Average Score

Correctness

1.67 / 5

Groundedness

1.58 / 5

Helpfulness

1.33 / 5

Overall

1.33 / 5

The LLM-judged examples were compared with the 15-example human review set
using the customer message as the matching key.

One example appeared in both evaluation sets. The LLM judge and human
reviewer both assigned an overall score of 2/5, giving 100% exact agreement
on this single overlapping example.

This agreement result is only a sanity check, not a statistically strong
measure of evaluator agreement, because the overlap contains only one
example.

The LLM evaluation was limited to 12 completed examples because the
Gemini free-tier request quota was reached. The full judge implementation
is available in:

scripts/llm_judge.py

---

# 11. Top 5 Failure Modes

The evaluation produced 128 incorrect predictions out of 200 examples.

The main failure modes were:

## 1. `other` → `damaged_or_missing_items`

This was the largest confusion pair.

Many broad or ambiguous customer complaints contain words associated with missing/damaged products, causing the classifier to select this intent.

## 2. `delivery_delayed` → `package_not_received`

These two intents are naturally similar.

A customer may say that a package has not arrived without clearly distinguishing between a delayed package and a package that was marked delivered but never received.

## 3. `device_issue` → `other`

Device-related complaints can be highly varied.

Without enough labelled examples, the classifier sometimes falls back to the broad `other` category.

## 4. `prime_video_issue` → `other`

Prime Video issues have relatively few training examples.

The classifier therefore struggles to learn a strong boundary between video-specific problems and general complaints.

## 5. `payment_issue` → `other`

Payment complaints can use many different words and descriptions.

The limited number of labelled examples makes it difficult for the classifier to recognize every payment-related phrasing.

---

# 12. What Is Misleading About My Headline Number?

The most tempting headline number is the **96.5% retrieval coverage**.

However, this number is misleading if presented as an accuracy metric.

The retrieval corpus and evaluation examples come from the same underlying dataset. Therefore, many evaluation messages have very similar historical examples available to the retriever.

Also, retrieval coverage only indicates that a sufficiently similar historical example was found. It does not prove that:

- the retrieved example is the correct solution,

- the generated reply is correct,

- the customer issue was actually resolved,

- or the classifier predicted the correct intent.

Therefore, the more meaningful model-quality headline is the intent classification result:

**36.0% accuracy and 23.94% macro F1 on the 200-example golden set.**

Even this number has limitations because the golden set was manually labelled from a limited sample rather than being a large stratified benchmark.

---

# 13. Example

Example customer message:

> My package hasn't arrived yet and it was supposed to be delivered yesterday.

The system can classify this as:

`package_not_received`

It then retrieves similar historical AmazonHelp conversations.

Example historical resolution:

> What does the current tracking indicate? Also, who does the tracking indicate as carrier?

The generated response is not copied directly. Instead, the system produces a concise support response such as:

> Sorry your package hasn't arrived yet. Please check the latest tracking status and carrier information. If the tracking does not explain the delay, please contact Amazon support so the order can be investigated.

The escalation policy can then decide whether this issue should be handled automatically or sent to a human agent.

---

# 14. Repository Structure

```text

hiver-support-agent/

│

├── data/

│   ├── raw/

│   │   └── twcs/

│   │       └── twcs.csv

│   │

│   ├── processed/

│   │   ├── amazonhelp_conversations.csv

│   │   ├── amazonhelp_training.csv

│   │   ├── training_sample.csv

│   │   ├── training_clean.csv

│   │   ├── evaluation_results.csv

│   │   ├── failure_analysis.csv

│   │   └── human_review.csv

│   │

│   └── golden/

│       └── golden_set.csv

│

├── scripts/

│   ├── collect_amazon_conversations.py

│   ├── collect_amazon_training.py

│   ├── create_golden_set.py

│   ├── create_training_sample.py

│   ├── label_training_batch_1.py

│   ├── label_training_batch_2.py

│   ├── label_training_batch_3.py

│   ├── label_training_batch_4.py

│   ├── label_training_batch_5.py

│   ├── label_training_batch_6.py

│   ├── check_overlap.py

│   ├── remove_overlap.py

│   ├── train_baselines.py

│   ├── retrieval.py

│   ├── demo.py

│   ├── evaluate_agent.py

│   ├── analyze_failures.py

│   ├── create_human_review.py

│   └── llm_judge.py

│

├── tests/

├── notebooks/

├── requirements.txt

├── .gitignore

└── README.md



---



## 15. How to Run

### Step 1: Create Virtual Environment

```bash

python -m venv venv

```

On Windows:

```powershell

venv\Scripts\activate

```

### Step 2: Install Dependencies

```bash

pip install -r requirements.txt

```

### Step 3: Add the Dataset

Download the Kaggle **Customer Support on Twitter** dataset and place:

```text

twcs.csv

```

inside:

```text

data/raw/twcs/

```

The raw dataset is not included in the repository because of its large size.

### Step 4: Run Evaluation

The required processed training and golden files are included in the repository. Run:

```bash

python scripts/evaluate_agent.py

```

This generates:

```text

data/processed/evaluation_results.csv

```

The evaluation is designed to run in less than 15 minutes on a normal laptop.

### Step 5: Analyze Failures

```bash

python scripts/analyze_failures.py

```

This generates:

```text

data/processed/failure_analysis.csv

```

### Step 6: Run the Demo

```bash

python scripts/demo.py

```

Enter a customer message when prompted.

The system returns:

1. Predicted intent

2. Intent confidence

3. Similar historical conversations

4. Suggested reply

5. Auto-handle or escalation decision

6. Escalation reason when applicable

---

## 16. Reproducing the Main Results

The main headline classification results are:

| Metric | Result |

|---|---:|

| Accuracy | **36.0%** |

| Macro F1 | **23.94%** |

The baseline results are:

| Model | Accuracy | Macro F1 |

|---|---:|---:|

| Majority Class | 28.5% | 3.17% |

| TF-IDF + Logistic Regression | **36.0%** | **23.94%** |

The evaluation uses the manually labelled 200-example golden set.

The repository includes the processed training and historical retrieval data
required by the main evaluation, so a fresh clone does not require the full
3-million-tweet raw dataset to reproduce the headline classification results.

Once dependencies are installed, the main evaluation can be reproduced in
less than 15 minutes on a normal laptop.

Run:

python scripts/evaluate_agent.py

This generates:

data/processed/evaluation_results.csv

---

## 17. Demo

Run:

```bash

python scripts/demo.py

```

Example input:

```text

My package hasn't arrived yet and it was supposed to be delivered yesterday.

```

Example system behavior:

```text

Predicted intent:

package_not_received

```

The system then:

1. Predicts the customer intent.

2. Searches historical AmazonHelp conversations.

3. Retrieves similar customer issues and historical responses.

4. Generates a concise support reply.

5. Decides whether to auto-handle or escalate.

Example generated reply:

```text

Sorry your package hasn't arrived yet. Please check the latest

tracking status and carrier information. If the tracking does not

explain the delay, please contact Amazon support so the order can

be investigated.

```

The response is generated from an intent-specific support template and informed by historical AmazonHelp resolutions. It does not directly copy the historical response.

---

## 18. One-Week Improvement Plan

### Day 1–2: Improve Labelling

Create a larger and more balanced golden evaluation set.

Give additional labelled examples to low-frequency intents such as:

- `order_change`

- `order_cancellation`

- `content_issue`

- `prime_video_issue`

### Day 3: Improve Intent Boundaries

Review the most confused intent pairs:

- `delivery_delayed` vs `package_not_received`

- `payment_issue` vs `prime_billing`

- `device_issue` vs `other`

Create clearer annotation rules for these cases.

### Day 4: Improve Retrieval Evaluation

Create a completely held-out retrieval corpus.

The evaluation examples should not come from the same underlying pool used for retrieval.

This would provide a more reliable measurement of retrieval quality.

### Day 5: Improve Reply Generation

Use the retrieved historical resolutions more directly to produce responses while ensuring that:

- customer-specific information is not copied,

- unrelated historical details are not included,

- the response remains concise and safe.

### Day 6: Improve Escalation

Test the escalation policy on high-risk cases such as:

- payment problems,

- account access,

- order cancellation,

- ambiguous customer requests.

Tune the confidence and similarity thresholds using a validation set.

### Day 7: Complete LLM Evaluation

Run the existing LLM-as-judge script with an available API budget.

Compare:

- human scores,

- LLM judge scores,

- classifier confidence,

- retrieval similarity.

This would provide stronger evidence for response quality and human/LLM agreement.

---

## 19. Decision Log

### Decision 1 — Selected AmazonHelp

**Reason:** AmazonHelp provides a large number of customer-support conversations covering different types of customer problems.

### Decision 2 — Defined 14 Intents

**Reason:** A small taxonomy makes the prototype easier to label, train, evaluate, and explain.

### Decision 3 — Used TF-IDF + Logistic Regression

**Reason:** It is fast, simple, interpretable, and suitable for a lightweight prototype.

### Decision 4 — Added a Majority-Class Baseline

**Reason:** It provides a simple lower-bound comparison for the classifier.

### Decision 5 — Created a 200-Example Golden Set

**Reason:** 200 manually labelled examples satisfy the required 150–250 evaluation-set size while keeping manual labelling feasible.

### Decision 6 — Created 500 Labelled Training Examples

**Reason:** This provides enough labelled data to build a working prototype without spending excessive time on manual annotation.

### Decision 7 — Removed 25 Exact Overlaps

**Reason:** Exact customer-message overlaps between training and golden data could artificially improve evaluation results.

### Decision 8 — Used TF-IDF Retrieval

**Reason:** It provides fast historical-example retrieval without requiring an external vector database or embedding service.

### Decision 9 — Retrieved Top 3 Historical Examples

**Reason:** Three examples provide useful historical context without adding unnecessary complexity.

### Decision 10 — Used Rule-Based Reply Generation

**Reason:** The objective was to produce grounded support replies quickly without training a generative language model.

### Decision 11 — Added Conservative Escalation

**Reason:** Payment, account-access, and cancellation requests can have higher consequences and should be handled conservatively.

### Decision 12 — Reported Retrieval Coverage Separately

**Reason:** Retrieval coverage is not equivalent to answer accuracy or successful customer resolution.

### Decision 13 — Performed Human Review

**Reason:** Human scoring provides additional evidence about the quality of generated replies beyond automated classification metrics.

### Decision 14 — Did Not Fabricate LLM-Judge Results

**Reason:** The LLM API account had no remaining credits during evaluation. The judge implementation is included, but unsupported scores are not reported.

---

## 20. Limitations

The current prototype has several limitations:

- The labelled training set is relatively small.

- Several intents contain relatively few training examples.

- Intent classification accuracy is only 36%.

- Macro F1 is only 23.94%, showing difficulty across minority classes.

- Retrieval evaluation is not fully independent because the retrieval and golden examples originate from the same underlying dataset.

- Retrieval coverage should therefore not be interpreted as retrieval accuracy.

- The reply generator is rule-based rather than a generative model.

- Human evaluation was performed on only 15 examples.

- LLM-as-judge evaluation was limited to 12 examples because of the Gemini free-tier request quota.

- Human-LLM agreement was checked on one overlapping example and showed 100% exact agreement on that example, but the sample is too small to support a strong agreement claim.

- The golden set is manually labelled but is not a large statistically stratified benchmark.

- The prototype is intended for experimentation and demonstration rather than direct production deployment.

---

## 21. Final Results Summary

| Component | Result |

|---|---:|

| Golden evaluation examples | **200** |

| Training examples before overlap removal | **500** |

| Exact overlaps removed | **25** |

| Final training examples | **475** |

| Majority baseline accuracy | **28.5%** |

| Majority baseline macro F1 | **3.17%** |

| TF-IDF + Logistic Regression accuracy | **36.0%** |

| TF-IDF + Logistic Regression macro F1 | **23.94%** |

| Retrieval coverage | **96.5%** |

| Average retrieval similarity | **0.965** |

| Auto-handle rate | **81%** |

| Escalation rate | **19%** |

| Human correctness score | **3.07 / 5** |

| Human groundedness score | **3.73 / 5** |

| Human helpfulness score | **3.60 / 5** |

| Human overall score | **3.93 / 5** |

### Overall

The prototype demonstrates the complete customer-support workflow:

**Classify → Retrieve → Draft → Decide**

The strongest result is that the system provides a complete, lightweight and reproducible support-agent pipeline using classical machine learning and historical AmazonHelp conversations.

The main weakness is intent classification performance, particularly for minority and closely related intents. The next iteration should therefore focus on better-balanced labelling, clearer intent definitions, independent retrieval evaluation, and stronger response-quality evaluation.

The reported results are intentionally conservative and include the limitations of the current experiment rather than overstating prototype performance.