import os
from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg')
stim1 = ['S  1', 'S  2', 'S  3', 'S  4', 'S  5', 'S  6', 'S  8', 'S  9']  # stimulus 1 markers
stim2 = ['S 65', 'S 66', 'S 67', 'S 68', 'S 69', 'S 70', 'S 72', 'S 73']  # stimulus 2 markers
response = ['S129', 'S130', 'S131', 'S132', 'S133', 'S134', 'S136', 'S138']  # response markers

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
        print(stim_mrk)
        if stim_mrk in stim1:
            print('stimulus marker is type 1')
            df.at[index, 'Stimulus Type'] = 'stim1'
        elif stim_mrk in stim2:
            print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 'stim2'
        elif stim_mrk in response:
            print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
        else:
            print('invalid stimulus marker')
            df.at[index, 'Stimulus Type'] = None




