import os
import filecmp
import numpy as np
import pandas as pd

"""
   Script to combine the info tables from BraTS20 and BraTS21 datasets so that
   the surival and resection status labels can be used for BraTS21
"""

def find_matching_subjects(path_a, path_b):
    matching_subjects = []

    # List all subject folders in datasets A and B
    subjects_a = [d for d in os.listdir(path_a) if os.path.isdir(os.path.join(path_a, d))]
    subjects_b = [d for d in os.listdir(path_b) if os.path.isdir(os.path.join(path_b, d))]

    # Iterate over each subject in dataset B
    for subject_b in subjects_b:
        path_to_t1_b = os.path.join(path_b, subject_b, f"{subject_b}_t1.nii.gz")

        # Iterate over each subject in dataset A
        for subject_a in subjects_a:
            path_to_t1_a = os.path.join(path_a, subject_a, f"{subject_a}_t1.nii.gz")

            # Compare the T1 scans
            if filecmp.cmp(path_to_t1_a, path_to_t1_b, shallow=False):
                matching_subjects.append((subject_a, subject_b))
                break  # No need to check other subjects in A

    return matching_subjects

# Example usage
path_to_dataset_a = '../brain_data/MICCAI_BraTS2020_TrainingData'
path_to_dataset_b = '../brain_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'
matching_pairs = find_matching_subjects(path_to_dataset_a, path_to_dataset_b)
pairs_a, pairs_b = [i[0] for i in matching_pairs], [i[1] for i in matching_pairs]
print(matching_pairs)

info_2020 = os.path.join(path_to_dataset_a, 'survival_info.csv')
info_2021 = '../brain_data/train_labels.csv'

df20 = pd.read_csv(info_2020)
df21 = pd.read_csv(info_2021)

subjects_rsna = os.listdir(path_to_dataset_b)

subject_id, mgmt, rs, age, survival = [], [], [], [], []
df = pd.DataFrame()
for i in range(len(subjects_rsna)):
    id21 = int(subjects_rsna[i].split('_')[-1])
    id20 = pairs_a[pairs_b.index(subjects_rsna[i])] if subjects_rsna[i] in pairs_b else None
    
    subject_id.append(id21)
    mgmt.append(df21['MGMT_value'][df21['BraTS21ID']==id21].values[0] if id21 in df21['BraTS21ID'].values else np.nan)
    rs.append(df20['Extent_of_Resection'][df20['Brats20ID']==id20].values[0] if id20 in df20['Brats20ID'].values else np.nan)
    age.append(df20['Age'][df20['Brats20ID']==id20].values[0] if id20 in df20['Brats20ID'].values else np.nan)
    survival.append(df20['Survival_days'][df20['Brats20ID']==id20].values[0] if id20 in df20['Brats20ID'].values else np.nan)

df['BraTS21ID'] = subject_id
df['MGMT_value'] = mgmt
df['Extent_of_Resection'] = rs
df['Age'] = age
df['Survival_days'] = survival

df.to_csv('BraTS2021_complete_table.csv', index=False)