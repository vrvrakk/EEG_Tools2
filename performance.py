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
    if files.endswith('ele.vmrk'):
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
    # print(df_name)
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk in stim1.values():
            # print('stimulus marker is type 1')
            df.at[index, 'Stimulus Type'] = 'stim1'
            df.at[index, 'Numbers'] = next(key for key, value in stim1.items() if value == stim_mrk)
        elif stim_mrk in stim2.values():
            # print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 'stim2'
            df.at[index, 'Numbers'] = next(key for key, value in stim2.items() if value == stim_mrk)
        elif stim_mrk in response.values():
            # print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
            df.at[index, 'Numbers'] = next(key for key, value in response.items() if value == stim_mrk)
        else:
            # print('invalid stimulus marker')
            df.at[index, 'Stimulus Type'] = None
            df.at[index, 'Numbers'] = 0
    df['Numbers'] = df['Numbers'].astype(int)
    df['Position'] = df['Position'].astype(int)

# discarding None values:
for df_name, df in dfs.items():
    rows_to_remove = []
    for index, marker in enumerate(df['Stimulus Type']):
        if marker is None:
            rows_to_remove.extend([index, index+1])
    df.drop(rows_to_remove, inplace=True)
    df.reset_index(drop=True, inplace=True)

# remove unwanted rows from 1 dataset:
discard_rows = list(range(1802, 1806))
dfs['df_240215_hz_azimuth'] = dfs['df_240215_hz_azimuth'].drop(discard_rows)

# create new df with stim1 and responses:
responses_dfs = {}
for df_name, df in dfs.items():
    # Filter rows where the stimulus type is 'response' or 'stim1'
    responses_df = df[(df['Stimulus Type'] == 'response') | (df['Stimulus Type'] == 'stim1')]
    responses_df.reset_index(drop=True, inplace=True)
    responses_dfs[df_name] = responses_df


# find correct responses:
performance_df = {}
for df_name, df in dfs.items():
    correct_responses_count = 0
    misses = 0
    false_responses_count = 0
    distractor_responses = 0
    RTs = []

    previous_stim1_number = None
    for i in range(len(df)-1, -1, -1):  # iterate in reverse through df
        if df.at[i, 'Stimulus Type'] == 'response':
            response_number = df.at[i, 'Numbers']  # get response number
            response_timestamp = pd.to_numeric(df.at[i, 'Position'])
            # Look for the previous 'stim1' stimulus
            j = i - 1
            while j >= 0 and df.at[j, 'Stimulus Type'] != 'stim1':
                j -= 1
            if j >= 0:
                previous_stim1_number = df.at[j, 'Numbers']  # get previous 'stim1' number
                previous_stim1_timestamp = pd.to_numeric(df.at[j, 'Position'])
                print(f"Response to stimulus 1: {previous_stim1_number} was {response_number}")
                if response_number == previous_stim1_number:
                    correct_responses_count += 1
                    reaction_time_s1 = response_timestamp - previous_stim1_timestamp
                    RTs.append(reaction_time_s1)
                else:
                    false_responses_count += 1
                    reaction_time_false = response_timestamp - previous_stim1_timestamp
                    RTs.append(reaction_time_false)
                    for k in range(j, i):
                        if df.at[k, 'Stimulus Type'] == 'stim2' and response_number == df.at[k, 'Numbers']:
                            previous_stim2_timestamp = pd.to_numeric(df.at[k, 'Position'])
                            distractor_responses += 1
                            reaction_time_s2 = response_timestamp - previous_stim2_timestamp
                            RTs.append(reaction_time_s2)
                            print(f'Response to stimulus 2: {response_number} occurred ')
            else:
                print(f"No previous 'stim1' found for response {response_number}")
                misses += 1
    performance_df[df_name] = {'correct_responses': correct_responses_count,
                               'misses': misses,
                               'false_responses': false_responses_count,
                               'distractor_responses': distractor_responses,
                               'reaction_times': RTs}

mean_rt = int(np.mean(RTs))
median_rt = int(np.median(RTs))
max_rt = int(np.max(RTs))
min_rt = int(np.min(RTs))







