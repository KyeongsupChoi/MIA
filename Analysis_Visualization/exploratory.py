import pandas as pd

reporty = pd.read_csv('../Data/neumo_dataset.tsv', sep='\t', header=0)

print(reporty.head())
