import os
from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg')
stim1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
stim2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers

# select .vmrk files:
marker_files = []
for files in os.listdir(data_path):
    if files.endswith('azimuth.vmrk'):
        marker_files.append(data_path / files)

# save marker files as pandas dataframe:
columns = ['Stimulus Stream', 'Position']
dfs = {}
for index, file_info in enumerate(marker_files):
    file_name = file_info.stem
    df = pd.read_csv(file_info, delimiter='\t', header=None) #
    df_name = f'df_{file_name}'
    df = df.iloc[10:]
    df = df.reset_index(drop=True, inplace=False)
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x)
    df = df.iloc[:, 1:3]
    df.insert(0, 'Stimulus Type', None)
    df.insert(2, 'Numbers', None)
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns [1]]
    dfs[df_name] = df

# define stimulus types:
for df_name, df in dfs.items():
    print(df_name)
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk in stim1.values():
            print('stimulus marker is type 1')
            df.at[index, 'Stimulus Type'] = 'stim1'
            df.at[index, 'Numbers'] = next(key for key, value in stim1.items() if value == stim_mrk)
        elif stim_mrk in stim2.values():
            print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 'stim2'
            df.at[index, 'Numbers'] = next(key for key, value in stim2.items() if value == stim_mrk)
        elif stim_mrk in response.values():
            print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
            df.at[index, 'Numbers'] = next(key for key, value in response.items() if value == stim_mrk)
        else:
            print('invalid stimulus marker')
            df.at[index, 'Stimulus Type'] = None
            df.at[index, 'Numbers'] = 0
    df['Numbers'] = df['Numbers'].astype(int)

# discarding None values:
for df_name, df in dfs.items():
    rows_to_remove = []
    for index, marker in enumerate(df['Stimulus Type']):
        if marker is None:
            print(index)
            rows_to_remove.extend([index, index+1])
    df.drop(rows_to_remove, inplace=True)

# calculate correct responses:
# TODO calculate error rate


