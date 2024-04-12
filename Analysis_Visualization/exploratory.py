import pandas as pd

# Define columns of interest
cols = ["ImageID", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Pediatric",
        "Labels", "group", "Partition", "Subject_occurrences",
        "Partition_occurrences", "Partitionlabel_occurrences"]

# Read the data from the TSV file into a DataFrame
reporty = pd.read_csv('../data/neumo_dataset_balanced_0.tsv', usecols=cols, sep='\t')

print(reporty[reporty["Labels"].str.contains('consolidation')].to_string())

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
'''
286804640197287451994953789098892695021_fbtbre.png, ['normal']
286928480601056410982743781154307721183_smnwmy.png, ['normal']
288495754523083562887862469328983763679_lqlhs0.png, ['normal']
294277257517832662316898718773785524416_ediicb.png, ['consolidation', 'pneumonia']
3527843935424277424025879480511614056_6i9e4v.png, ['consolidation', 'pneumonia']
68031809687808465969241796432987316467_wir071.png, ['consolidation', 'pneumonia']
69482454427136223031415032657955895290_nw1zi4.png, ['normal']
69977278055621668879692049606137189503_71qdfh.png, ['normal']
70043412391326422547450800096657522195_bcz889.png, ['normal']
70095835523670596747545529863919032540_g4a8wa.png, ['normal']
70107197908653175536461660700427492531_i8iwjz.png, ['normal']
70226522249036129214265554497929105429_w8esew.png, ['normal']
70590232516607663751215182003452348392_xsgtpp.png, ['normal']
216840111366964013590140476722013025090537801_02-007-190.png, ['consolidation', ' pneumonia']
216840111366964013590140476722013025090617613_02-008-046.png, ['consolidation', ' pneumonia']
216840111366964013534861372972012345121259946_01-132-177.png, ['consolidation', ' pneumonia'] 
216840111366964013076187734852011234081129294_00-117-155.png,  ['consolidation', ' pneumonia'] 
216840111366964013076187734852011234082333431_00-118-020.png, ['consolidation', ' pneumonia']
216840111366964013076187734852011242142347283_00-114-144.png, ['consolidation', ' pneumonia'] 
70689082589328298444485748895432333078_hd412x.png, ['normal']'''