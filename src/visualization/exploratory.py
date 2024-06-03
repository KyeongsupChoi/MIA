import pandas as pd

allevery = pd.read_csv('../../data/raw/neumo_dataset.tsv', sep='\t')

print(allevery.head().to_string())

print(allevery.describe().to_string())

print(allevery.shape)

print(allevery.columns)

print(allevery.info())

print(allevery['StudyDate_DICOM'].value_counts())

# The date the study was conducted and Chest Xray was recorded, max is 20150120: 59

print(allevery['StudyID'].value_counts())

# ID for each study, 126022968388682456059208259745221627283: 12 is max. Incongruous with study date. Perhaps multiple studies done on same date

for xid in allevery.columns:
        print(allevery[xid].value_counts())

# Define columns of interest
cols = ["ImageID", "PatientID", "PatientBirth",
        "Projection", "Pediatric", "Modality_DICOM",
        "Manufacturer_DICOM", "Labels", "group"]

# Read the data from the TSV file into a DataFrame
reporty = pd.read_csv('../../data/raw/neumo_dataset.tsv', usecols=cols, sep='\t')

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