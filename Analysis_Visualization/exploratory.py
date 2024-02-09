import pandas as pd

cols = ["ImageID", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Pediatric",
        "Labels", "group", "Partition", "Subject_occurrences",
        "Partition_occurrences", "Partitionlabel_occurrences"]

reporty = pd.read_csv('../Data/neumo_dataset_balanced_0.tsv', usecols=cols, sep='\t')

df_exploded = reporty.explode('Labels')

# Count the occurrences of each value
value_counts = df_exploded['Labels'].value_counts()

print(value_counts[0:10])


# ['normal']                            9004
# ['pneumonia']                          783
# ['infiltrates']                        739
# ['infiltrates', ' pneumonia']          478
# ['consolidation', ' pneumonia']        210
# ['consolidation']                      4

sliced = reporty[reporty['Labels'].isin(["['normal']", "['consolidation', ' pneumonia']"])]
print(len(sliced))
sliced.to_csv("sliced.csv", encoding='utf-8', index=False)