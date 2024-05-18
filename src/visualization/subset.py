import pandas as pd
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# Load the TSV file into a pandas DataFrame
df = pd.read_csv("../../data/raw/neumo_dataset_balanced_0.tsv", delimiter="\t")

# Subset the DataFrame for rows where the label column contains ['normal']
# Subset the DataFrame for rows where the label column contains only ['pneumonia']

df = df[df['Projection'].apply(lambda x: 'PA' in x)]

df = df[df['Pediatric'].apply(lambda x: 'No' in x)]
subset_df = df[df['Labels'].apply(lambda x: 'normal' in x)]

# Take the first 200 rows from the subset
final_subset = subset_df.head(200)


subset_df = df[df['Labels'].apply(lambda x: len(x) == 13 and 'pneumonia' in x)]

subset_df = subset_df.head(200)

# Take the first 200 rows from the subset using .loc, if available

final = pd.concat([final_subset, subset_df], axis=0)

print(final.to_string())
print(len(final))

final.to_csv('exporty.csv')
"""
import shutil
import os

# 원래 폴더와 대상 폴더 경로 정의
source_folder = "../data/img/"
target_folder = "../data/exporty"

# 대상 폴더가 존재하지 않는 경우 생성
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 이미지 파일 목록 가져오기
image_files = final['ImageID'].values

# 각 이미지 파일을 대상 폴더로 이동
for image_file in image_files:
    source_path = os.path.join(source_folder, image_file)
    target_path = os.path.join(target_folder, image_file)
    shutil.copyfile(source_path, target_path)"""
