import pandas as pd

input_file = "data/processed/amazonhelp_training.csv"
output_file = "data/processed/training_sample.csv"

df = pd.read_csv(input_file)

# Random sample for manual labeling
sample = df.sample(n=500, random_state=42)

# Add empty intent column
sample["intent"] = ""

sample.to_csv(output_file, index=False)

print("Training sample created!")
print("Number of examples:", len(sample))
print("Saved to:", output_file)