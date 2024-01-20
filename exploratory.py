import pandas as pd

reporty = pd.read_csv('neumo_dataset.tsv', sep='\t', header=0)

print(reporty.to_string())
