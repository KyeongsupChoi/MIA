import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv("../data/sliced.csv")

# Split the dataset into training and test sets based on the 'Labels' column
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Labels'], random_state=42)

# Display the shapes of the training and test sets
print("Training dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

# Save the split datasets to new CSV files
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)