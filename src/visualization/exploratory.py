import pandas as pd

# Read the entire dataset from a TSV file into a DataFrame named 'allevery'
allevery = pd.read_csv('../../data/raw/neumo_dataset.tsv', sep='\t')

# Print the first few rows of the dataset to get an initial look at the data
print(allevery.head().to_string())

# Print the summary statistics of the dataset to understand its numerical properties
print(allevery.describe().to_string())

# Print the shape of the dataset (number of rows and columns)
print(allevery.shape)

# Print the column names of the dataset
print(allevery.columns)

# Print detailed information about the dataset (data types, non-null counts, etc.)
print(allevery.info())

# Loop through each column in the dataset and print the frequency of each unique value
for xid in allevery.columns:
    print(allevery[xid].value_counts())

# Define a list of columns of interest
cols = ["ImageID", "PatientID", "PatientBirth",
        "Projection", "Pediatric", "Modality_DICOM",
        "Manufacturer_DICOM", "Labels", "group"]

# Read only the specified columns from the TSV file into a DataFrame named 'reporty'
reporty = pd.read_csv('../../data/raw/neumo_dataset.tsv', usecols=cols, sep='\t')

# Separate multiple labels in the 'Labels' column into individual rows
df_exploded = reporty.explode('Labels')

# Count the occurrences of each unique label
value_counts = df_exploded['Labels'].value_counts()

# Print the top 10 most frequent labels
print(value_counts[0:10])

# Filter the DataFrame to include only rows with specific labels
sliced = reporty[reporty['Labels'].isin(["['normal']", "['consolidation', ' pneumonia']"])]

# Print the filtered DataFrame
print(sliced)

# (Optional) Write the filtered DataFrame to a CSV file
# Uncomment the following line to save the sliced DataFrame to a CSV file
#sliced.to_csv("sliced.csv", encoding='utf-8', index=False)

"""
Example output of label counts:
Labels
['normal']                            9004
['pneumonia']                          783
['infiltrates']                        739
['infiltrates', ' pneumonia']          478
['consolidation', ' pneumonia']        210
['COPD signs']                         156
['alveolar pattern', ' pneumonia']     156
['scoliosis']                          137
['infiltrates', 'unchanged']           134
['alveolar pattern',"""