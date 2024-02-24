import os
from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg')
stim1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
stim2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
# invalid_markers = ['S 75', 'S195', 'S 64', 'S135', 'S138', 'S139', 'S140', 'S141', 'S142', 'S198', 'S199', 'S200',
#                  'S201', 'S202',
#                   'S203', 'S204', 'S205', 'S206', 'S207', 'S197']
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers


# select .vmrk files:
marker_files = []
for files in os.listdir(data_path):
    if files.endswith('azimuth.vmrk'):
        marker_files.append(data_path / files)

# save marker files as pandas dataframe:
columns = ['Stimulus Stream', 'Position', 'Time Difference']
dfs = {}
for index, file_info in enumerate(marker_files):
    file_name = file_info.stem
    df = pd.read_csv(file_info, delimiter='\t', header=None)  # t for tabs
    df_name = f'df_{file_name}'
    df = df.iloc[10:]
    df = df.reset_index(drop=True, inplace=False)
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x)
    df = df.iloc[:, 1:3]
    df.insert(0, 'Stimulus Type', None)
    df.insert(2, 'Numbers', None)
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]]
    df['Time'] = np.nan
    df['Time Differences'] = np.nan
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

# discard None values (pray no shit shifted around):
for df_name, df in dfs.items():
    rows_to_drop = []
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk not in stim1.values() and stim_mrk not in stim2.values() and stim_mrk not in response.values():
            rows_to_drop.append(index)
    # Drop the marked rows from the DataFrame
    df.drop(rows_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

# remove Stimulus Stream, convert Numbers and Positions vals to int:
dfs_updated = {}
for df_name, df in dfs.items():
    df['Numbers'] = pd.to_numeric(df['Numbers'])
    df['Position'] = pd.to_numeric(df['Position'])
    df['Time Differences'] = pd.to_numeric((df['Time Differences']))
    df['Time'] = df['Position'] / 500
    df.drop(columns=['Stimulus Stream'], inplace=True)
    # drop response rows that are invalid:
    drop_responses = []
    for index in range(len(df)-1):
        current_timestamp = df.at[index, 'Position'] / 500
        next_timestamp = df.at[index + 1, 'Position'] / 500
        time_difference = next_timestamp - current_timestamp
        df.at[index, 'Time Differences'] = time_difference
    for index, stimulus in enumerate(df['Stimulus Type']):
        if stimulus == 'response' and index < len(df) - 2:
            current_response = df.at[index, 'Numbers']
            next_response = df.at[index + 1, 'Numbers']
            over_next_response = df.at[index + 2, 'Numbers']
            if current_response == over_next_response:
                drop_responses.extend([index + 1, index + 2])
            elif current_response == next_response:
                drop_responses.append(index)
    df = df.drop(drop_responses)
    df.reset_index(drop=True, inplace=True)
    dfs_updated[df_name] = df

# TODO: calculate S1 responses, S2 responses, no responses, errors.

s1_responses = 0
for df_name, df in dfs_updated.items():
    for i, stimulus in enumerate(df['Stimulus Type']):
        if stimulus == 'stim1' and i + 1 < len(df):  # Check if the current stimulus is 'stim1' and there's a next stimulus
            next_stimulus = df.at[i + 1, 'Stimulus Type']
            stimulus_number = df.at[i, 'Numbers']
            while next_stimulus != 'stim1':
                if next_stimulus == 'response':  # Check if the next stimulus is 'response'
                    response_number = df.at[i + 1, 'Numbers']
                    if stimulus_number == response_number:  # Check if the response matches the stimulus number
                        s1_responses += 1
                    break  # Exit the loop once a response is found
                i += 1  # Move to the next stimulus
                if i + 1 < len(df):
                    next_stimulus = df.at[i + 1, 'Stimulus Type']
                else:
                    break  # Exit the loop if we reach the end of the DataFrame

s2_responses = 0
for df_name, df in dfs_updated.items():
    for i, stimulus in enumerate(df['Stimulus Type']):
        if stimulus == 'stim2' and i + 1 < len(df):  # Check if the current stimulus is 'stim1' and there's a next stimulus
            next_stimulus = df.at[i + 1, 'Stimulus Type']
            stimulus_number = df.at[i, 'Numbers']
            while next_stimulus != 'stim2':
                if next_stimulus == 'response':  # Check if the next stimulus is 'response'
                    response_number = df.at[i + 1, 'Numbers']
                    if stimulus_number == response_number:  # Check if the response matches the stimulus number
                        s2_responses += 1
                    break  # Exit the loop once a response is found
                i += 1  # Move to the next stimulus
                if i + 1 < len(df):
                    next_stimulus = df.at[i + 1, 'Stimulus Type']
                else:
                    break  # Exit the loop if we reach the end of the DataFrame

no_s1_response = []
no_s2_response = []

for df_name, df in dfs_updated.items():
    s1_indices = df.index[df['Stimulus Type'] == 'stim1']  # Get indices of 'stim1' onsets
    s2_indices = df.index[df['Stimulus Type'] == 'stim2']  # Get indices of 'stim2' onsets

    for s1_index in s1_indices:
        next_s1_index = s1_indices[s1_indices > s1_index].min()  # Find the index of the next 'stim1'
        responses_between_s1 = df.loc[s1_index:next_s1_index - 1]['Stimulus Type'] == 'response'
        if not responses_between_s1.any():
            no_s1_response.append((df_name, s1_index, next_s1_index))

    for s2_index in s2_indices:
        next_s2_index = s2_indices[s2_indices > s2_index].min()  # Find the index of the next 'stim2'
        responses_between_s2 = df.loc[s2_index:next_s2_index - 1]['Stimulus Type'] == 'response'
        if not responses_between_s2.any():
            no_s2_response.append((df_name, s2_index, next_s2_index))

errors_s1 = 0

for df_name, df in dfs_updated.items():
    s1_indices = df.index[df['Stimulus Type'] == 'stim1']  # Get indices of 'stim1' onsets
    s1_numbers = df.loc[s1_indices]['Numbers']

    for i, s1_index in enumerate(s1_indices):
        if i < len(s1_indices) - 1:
            next_s1_index = s1_indices[i + 1]  # Find the index of the next 'stim1'
        else:
            next_s1_index = len(df)

        response_between_s1 = df.loc[s1_index:next_s1_index]['Stimulus Type'] == 'response'

        if not response_between_s1.empty and not s1_numbers.empty:
            if not any(response_between_s1) and df.loc[s1_index]['Numbers'] != s1_numbers.iloc[i]:
                errors_s1 += 1

errors_s2 = 0
for df_name, df in dfs_updated.items():
    s2_indices = df.index[df['Stimulus Type'] == 'stim1']  # Get indices of 'stim1' onsets
    s2_numbers = df.iloc[s2_indices, 'Numbers']
    for i, s2_index in enumerate(s2_indices):
        if i < len(s2_indices) - 1:
            next_s2_index = s2_indices[i + 1]  # Find the index of the next 'stim1'
        else:
            next_s2_index = len(df)
        response_between_s2 = df.loc[s2_index:next_s2_index - 1, 'Stimulus Type'] == 'response'
        if not response_between_s2.empty and not any(response_between_s2) and df.loc[s2_index, 'Numbers'] != s2_numbers.iloc[i]:
            errors_s2 += 1


total_responses = errors + s1_responses + s2_responses + no_s1_response + no_s2_response

actual_responses = 0

for df_name, df in dfs_updated.items():
    response_count = df[df['Stimulus Type'] == 'response'].shape[0]
    actual_responses += response_count

print("Total number of responses from all DataFrames:", actual_responses)