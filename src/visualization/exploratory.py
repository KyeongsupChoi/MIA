import pandas as pd

# Define columns of interest
cols = ["ImageID", "PatientID", "PatientBirth",
        "Projection", "Pediatric", "Modality_DICOM",
        "Manufacturer_DICOM", "Labels", "group"]

# Read the data from the TSV file into a DataFrame
reporty = pd.read_csv('../../data/raw/neumo_dataset.tsv', usecols=cols, sep='\t')

print(reporty.describe())

print(reporty.shape)

print(reporty.columns)

print(reporty.info())

print(reporty.head().to_string())

print(reporty['Projection'].value_counts())

# Explode the 'Labels' column to separate multi-label entries
df_exploded = reporty.explode('Labels')

# Count the occurrences of each label
value_counts = df_exploded['Labels'].value_counts()

# Print the top 10 most frequent labels
print(value_counts[0:10])

# Slice the DataFrame to include only rows with specific labels
sliced = reporty[reporty['Labels'].isin(["['normal']", "['consolidation', ' pneumonia']"])]

# Print the length of the sliced DataFrame
print(sliced)

# Write the sliced DataFrame to a CSV file
#sliced.to_csv("sliced.csv", encoding='utf-8', index=False)

"""
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
['alveolar pattern', 'pneumonia']      125"""