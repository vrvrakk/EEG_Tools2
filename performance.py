import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


s1_files = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/s1')
s2_files = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/s2')
stim1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
stim2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers


# select .vmrk files:
marker_files = []
for files in os.listdir(s1_files):
    if files.endswith('azimuth.vmrk'):
        marker_files.append(s1_files / files)

# save marker files as pandas dataframe:
columns = ['Stimulus Stream', 'Position', 'Time Difference']
dfs = {}
for index, file_info in enumerate(marker_files):
    file_name = file_info.stem
    df = pd.read_csv(file_info, delimiter='\t', header=None)  # t for tabs
    df_name = f'df_{file_name}'
    df = df.iloc[10:] # delete first 10 rows
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
for df_name, df in dfs.items():
    df['Numbers'] = pd.to_numeric(df['Numbers'])
    df['Position'] = pd.to_numeric(df['Position'])
    df['Time Differences'] = pd.to_numeric((df['Time Differences']))
    df['Time'] = df['Position'] / 500
    df.drop(columns=['Stimulus Stream'], inplace=True)
    # drop response rows that are invalid:
    for index in range(len(df)-1):
        if df.at[index, 'Stimulus Type'] == 'response':
            current_timestamp = df.at[index, 'Position'] / 500
            next_timestamp = df.at[index + 1, 'Position'] / 500
            time_difference = next_timestamp - current_timestamp
            df.at[index, 'Time Differences'] = time_difference

# clear invalid responses:
dfs_updated = {}
drop_responses_dict = {}
time_differences_dict = {}
for df_name, df in dfs.items():
    time_differences_list = []
    drop_responses = []

    for index, stimulus in enumerate(df['Stimulus Type']):
        if stimulus == 'response' and index < len(df) - 2:
            current_response = df.at[index, 'Numbers']
            next_stimulus = df.at[index + 1, 'Stimulus Type']
            next_response = df.at[index + 1, 'Numbers']
            over_next_stimulus = df.at[index + 2, 'Stimulus Type']
            over_next_response = df.at[index + 2, 'Numbers']

            if next_stimulus == 'response' and current_response == next_response:
                time_differences_list.append(df.at[index, 'Time Differences'])
                drop_responses.append(index)
            elif over_next_stimulus == 'response' and current_response == over_next_response:
                time_differences_list.extend(
                    [df.at[index + 1, 'Time Differences'], df.at[index + 2, 'Time Differences']])
                drop_responses.extend([index + 1, index + 2])

    drop_responses_dict[df_name] = drop_responses # save invalid responses in dict
    time_differences_dict[df_name] = time_differences_list
    df = df.drop(drop_responses)
    df.reset_index(drop=True, inplace=True)
    dfs_updated[df_name] = df  # save updated dfs in dict


# TODO: calculate S1 responses, S2 responses, no responses, errors.
s1_responses_dict = {}
s2_responses_dict = {}
errors_dict = {}
misses_dict = {}
for df_name, df in dfs_updated.items():
    s1_responses = []
    s2_responses = []
    errors = []
    misses = []
    s1_indices = df.index[df['Stimulus Type'] == 'stim1']
    response_indices = df.index[df['Stimulus Type'] == 'response']
    for s1_index in s1_indices:
        s1_number = df.at[s1_index, 'Numbers']  # get corresponding number of stim1
        s1_time = df.at[s1_index, 'Time'] # get corresponding stim1 time
        next_s1_index = s1_indices[s1_indices > s1_index].min()  # get next s1 index
        window_data = df.loc[s1_index:next_s1_index - 1]

        # Find the corresponding number of 'stim2' within the window
        s2_numbers_within_window = window_data.loc[window_data['Stimulus Type'] == 'stim2', 'Numbers'].values
        s2_indices_within_window = window_data.index[window_data['Stimulus Type'] == 'stim2']

        responses_within_window = window_data[window_data['Stimulus Type'] == 'response']
        if len(responses_within_window) == 0:
            stimulus_type_missed = window_data['Stimulus Type'].values[0]

            stimulus_index_missed = window_data.index[0]

            stimulus_number_missed = window_data.at[stimulus_index_missed, 'Numbers']

            misses.append(('miss', window_data.at[stimulus_index_missed, 'Time'], stimulus_type_missed,
                           stimulus_number_missed, stimulus_index_missed))
        else:
            for response_index, response_row in responses_within_window.iterrows():
                response_number = response_row['Numbers']
                if response_number == s1_number:
                    s1_responses.append((response_index, 's1', response_number, response_row['Time'], 'stim1',
                                         s1_number, s1_index, s1_time))
                elif response_number in s2_numbers_within_window:
                    matching_s2_indices = s2_indices_within_window[s2_numbers_within_window == response_number]
                    for s2_index_within_window in matching_s2_indices:
                        s2_responses.append((response_index, 's2', response_number, response_row['Time'], 'stim2',
                                             window_data.at[s2_index_within_window, 'Numbers'],
                                             s2_index_within_window, window_data.at[s2_index_within_window, 'Time']))
                else:
                    stimulus_index_error = window_data.index[0]
                    stimulus_type_error = window_data.at[stimulus_index_error, 'Stimulus Type']
                    stimulus_number_error = window_data.at[stimulus_index_error, 'Numbers']
                    stimulus_time_error = window_data.at[stimulus_index_error, 'Time']
                    errors.append((response_index, 'error', response_number, response_row['Time'],
                                   stimulus_type_error, stimulus_index_error,
                                   stimulus_number_error, stimulus_time_error))

    s1_responses_dict[df_name] = {'s1_responses': s1_responses}
    s2_responses_dict[df_name] = {'s2_responses': s2_responses}
    errors_dict[df_name] = {'errors': errors}
    misses_dict[df_name] = {'misses': misses}

# for s2 as target:
# s1_responses_dict = {}
# s2_responses_dict = {}
# errors_dict = {}
# misses_dict = {}
# for df_name, df in dfs_updated.items():
#     s1_responses = []
#     s2_responses = []
#     errors = []
#     misses = []
#     s2_indices = df.index[df['Stimulus Type'] == 'stim2']
#     response_indices = df.index[df['Stimulus Type'] == 'response']
#     for s2_index in s2_indices:
#         s2_number = df.at[s2_index, 'Numbers']  # get corresponding number of stim2
#         s2_time = df.at[s2_index, 'Time'] # get corresponding stim2 time
#         next_s2_index = s2_indices[s2_indices > s2_index].min()  # get next s2 index
#         window_data = df.loc[s2_index:next_s2_index - 1]
#
#         # Find the corresponding number of 'stim1' within the window
#         s1_numbers_within_window = window_data.loc[window_data['Stimulus Type'] == 'stim1', 'Numbers'].values
#         s1_indices_within_window = window_data.index[window_data['Stimulus Type'] == 'stim1']
#
#         responses_within_window = window_data[window_data['Stimulus Type'] == 'response']
#         if len(responses_within_window) == 0:
#             stimulus_type_missed = window_data['Stimulus Type'].values[0]
#
#             stimulus_index_missed = window_data.index[0]
#
#             stimulus_number_missed = window_data.at[stimulus_index_missed, 'Numbers']
#
#             misses.append(('miss', window_data.at[stimulus_index_missed, 'Time'], stimulus_type_missed,
#                            stimulus_number_missed, stimulus_index_missed))
#         else:
#             for response_index, response_row in responses_within_window.iterrows():
#                 response_number = response_row['Numbers']
#                 if response_number == s2_number:
#                     s2_responses.append((response_index, 's2', response_number, response_row['Time'], 'stim2',
#                                          s2_number, s2_index, s2_time))
#                 elif response_number in s1_numbers_within_window:
#                     matching_s1_indices = s1_indices_within_window[s1_numbers_within_window == response_number]
#                     for s1_index_within_window in matching_s1_indices:
#                         s1_responses.append((response_index, 's1', response_number, response_row['Time'], 'stim1',
#                                              window_data.at[s1_index_within_window, 'Numbers'],
#                                              s1_index_within_window, window_data.at[s1_index_within_window, 'Time']))
#                 else:
#                     stimulus_index_error = window_data.index[0]
#                     stimulus_type_error = window_data.at[stimulus_index_error, 'Stimulus Type']
#                     stimulus_number_error = window_data.at[stimulus_index_error, 'Numbers']
#                     stimulus_time_error = window_data.at[stimulus_index_error, 'Time']
#                     errors.append((response_index, 'error', response_number, response_row['Time'],
#                                    stimulus_type_error, stimulus_index_error,
#                                    stimulus_number_error, stimulus_time_error))
#
#     s1_responses_dict[df_name] = {'s1_responses': s1_responses}
#     s2_responses_dict[df_name] = {'s2_responses': s2_responses}
#     errors_dict[df_name] = {'errors': errors}
#     misses_dict[df_name] = {'misses': misses}

# convert to dfs for readability

s1_responses_dfs = {}
for df_name, sub_dict in s1_responses_dict.items():

    rows = [{'response_index': row[0],
             'response_type': row[1],
             'response_number': row[2],
             'response_time': row[3],
             'stimulus_type': row[4],
             'stimulus_number': row[5],
             'stimulus_index': row[6],
             'stimulus_time': row[7]} for row in sub_dict['s1_responses']]

    # Create the DataFrame
    s1_responses_df = pd.DataFrame(rows)
    s1_responses_dfs[df_name] = s1_responses_df



# Convert s2_responses_dict
s2_responses_dfs = {}
for df_name, sub_dict in s2_responses_dict.items():

    rows = [{'response_index': row[0],
             'response_type': row[1],
             'response_number': row[2],
             'response_time': row[3],
             'stimulus_type': row[4],
             'stimulus_number': row[5],
             'stimulus_index': row[6],
             'stimulus_time': row[7]} for row in sub_dict['s2_responses']]

    s2_responses_df = pd.DataFrame(rows)
    s2_responses_dfs[df_name] = s2_responses_df


# Convert errors_dict
errors_dfs = {}
for df_name, sub_dict in errors_dict.items():

    rows = [{'response_index': row[0],
             'response_type': row[1],
             'response_number': row[2],
             'response_time': row[3],
             'stimulus_type': row[4],
             'stimulus_index': row[5],
             'stimulus_number': row[6],
             'stimulus_time': row[7]} for row in sub_dict['errors']]

    # Create the DataFrame
    errors_df = pd.DataFrame(rows)
    errors_dfs[df_name] = errors_df

# Convert misses_dict
misses_dfs = {}
for df_name, sub_dict in misses_dict.items():

    rows = [{'miss_type': row[0],
             'time_missed': row[1],
             'stimulus_type_missed': row[2],
             'stimulus_index_missed': row[3],
             'stimulus_number_missed': row[4]} for row in sub_dict['misses']]

    misses_df = pd.DataFrame(rows)
    misses_dfs[df_name] = misses_df


response_counts = {}

for df_name, df in dfs_updated.items():
    response_counts[df_name] = df['Stimulus Type'].value_counts().get('response', 0)

print(response_counts)


# Combine all s1_responses_dfs, s2_responses_dfs, errors_dfs, and misses_dfs
combined_s1_responses = pd.concat(s1_responses_dfs.values())
combined_s2_responses = pd.concat(s2_responses_dfs.values())
combined_errors = pd.concat(errors_dfs.values())
combined_misses = pd.concat(misses_dfs.values())

# Count each response type
s1_counts = combined_s1_responses['response_type'].value_counts()
s2_counts = combined_s2_responses['response_type'].value_counts()
error_counts = combined_errors['response_type'].value_counts()
miss_counts = combined_misses['miss_type'].value_counts()

# Combine counts into one DataFrame
combined_counts = pd.DataFrame({
    's1_responses': s1_counts,
    's2_responses': s2_counts,
    'errors': error_counts,
    'misses': miss_counts
}).fillna(0)  # Fill NaN values with 0 if any type is missing

total_responses = combined_counts.sum(axis=1)
grand_total = total_responses.sum()
percentage = (combined_counts / grand_total) * 100

# Plot the combined counts
plt.figure(figsize=(12, 8))
percentage.plot(kind='bar', stacked=False)
plt.title('Comparison of Responses, Errors, and Misses')
plt.xlabel('Response Type')
plt.ylabel('Count')
plt.xticks(rotation=45)  # rotation of the x labels
plt.legend(title='Response Category')
plt.show()

# fig, axis = plt.subplots(1,1)
# plt.figure(figsize=(12, 8))
# fig = combined_counts.plot(kind='bar', stacked=False)
# axis.set_title('Comparison of Responses, Errors, and Misses')
# axis.get_xticks()
# axis.set_xlabel('Response Type')
# plt.ylabel('Count')
# plt.set_xticks(rotation=45)
# plt.legend(title='Response Category')
# plt.show()


# TODO: calculate RT:
# stimulus_time - response_time
# s1:
RT_s1 = []
RT_s1_dfs = {}
for i, (df_name, df) in enumerate(s1_responses_dfs.items()):
    if not df.empty and 'stimulus_time' in df.columns:
        stim1_time = df['stimulus_time'].values
        response_s1 = df['response_time'].values
        RT = response_s1 - stim1_time
        RT_s1.append(RT)
        RT_s1_dfs[df_name] = pd.DataFrame({'RT': RT})

RT_vals_s1 = []
for df_name, RT_dfs in RT_s1_dfs.items():
    RT_vals_s1.extend(RT_dfs['RT'].values)
plt.hist(RT_vals_s1, color='skyblue', edgecolor='black')
plt.xlabel('Reaction Time')
plt.ylabel('Frequency')
plt.title('Reaction Times Distribution')
plt.show()

# median RT:
RT_median = np.median(RT_vals_s1)
RT_mean = np.mean(RT_vals_s1)
RT_max = np.max(RT_vals_s1)
RT_min = np.min(RT_vals_s1)

RT_s2 = []
RT_s2_dfs = {}
for i, (df_name, df) in enumerate(s2_responses_dfs.items()):
    if not df.empty and 'stimulus_time' in df.columns:
        stim2_time = df['stimulus_time'].values
        response_s2 = df['response_time'].values
        RT = response_s2 - stim2_time
        RT_s2.append(RT)
        RT_s2_dfs[df_name] = pd.DataFrame({'RT': RT})

RT_vals_s2 = []
for df_name, RT_dfs in RT_s2_dfs.items():
    RT_vals_s2.extend(RT_dfs['RT'].values)
plt.hist(RT_vals_s2, color='skyblue', edgecolor='black')
plt.xlabel('Reaction Time')
plt.ylabel('Frequency')
plt.title('Reaction Times Distribution')
plt.show()

# calculate RTs
RT_median = np.median(RT_vals_s2)
RT_mean = np.mean(RT_vals_s2)
RT_max = np.max(RT_vals_s2)
RT_min = np.min(RT_vals_s2)
