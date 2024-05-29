import pandas as pd

# Define columns of interest
cols = ["ImageID", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Pediatric",
        "Labels", "group", "Partition", "Subject_occurrences",
        "Partition_occurrences", "Partitionlabel_occurrences"]

# Read the data from the TSV file into a DataFrame
reporty = pd.read_csv('../../data/raw/neumo_dataset_balanced_0.tsv', usecols=cols, sep='\t')


# Explode the 'Labels' column to separate multi-label entries
df_exploded = reporty.explode('Labels')

# Count the occurrences of each label
value_counts = df_exploded['Labels'].value_counts()

# ['normal']                            9004
# ['pneumonia']                          783
# ['infiltrates']                        739
# ['infiltrates', ' pneumonia']          478
# ['consolidation', ' pneumonia']        210
# ['consolidation']                      4

# Print the top 10 most frequent labels
#print(value_counts[0:10])

# Slice the DataFrame to include only rows with specific labels
sliced = reporty[reporty['Labels'].isin(["['normal']", "['consolidation', ' pneumonia']"])]

# Print the length of the sliced DataFrame
#print(len(sliced))

# Write the sliced DataFrame to a CSV file
sliced.to_csv("sliced.csv", encoding='utf-8', index=False)
