import pandas as pd 
 
file_path = "data/golden/golden_set.csv" 
 
df = pd.read_csv(file_path) 
 
for i, row in df.iloc[175:200].iterrows():
 
    print("\n----------------------------------------") 
    print("ROW:", i + 1) 
    print("Customer:") 
    print(row["customer_message"]) 
 
    print("\nAmazonHelp reply:") 
    print(row["amazon_reply"])