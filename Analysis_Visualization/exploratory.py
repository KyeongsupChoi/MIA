import pandas as pd

cols = ["ImageID", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Pediatric",
        "Labels", "group", "Partition", "Subject_occurrences",
        "Partition_occurrences", "Partitionlabel_occurrences"]

reporty = pd.read_csv('../Data/neumo_dataset_balanced_0.tsv', usecols=cols, sep='\t')

print(reporty.to_string())
