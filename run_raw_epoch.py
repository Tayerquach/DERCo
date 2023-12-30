from utils.epochs import create_epochs_from_events
from utils.raw import read_raw_data, filter_raw

from pathlib import Path
import numpy as np
import os
import mne
from mne import EpochsArray
import argparse
from random import seed

def customise_events(arr):
    # Find the indices of consecutive elements with the same value in the third column
    consecutive_indices = []
    current_value = arr[0, 2]
    start_index = 0

    for i in range(1, len(arr)):
        if arr[i, 2] != current_value:
            if i - start_index > 1:
                consecutive_indices.extend(range(start_index + 1, i))
            start_index = i
            current_value = arr[i, 2]

    # Check if the last sequence also has consecutive elements
    if len(arr) - start_index > 1:
        consecutive_indices.extend(range(start_index + 1, len(arr)))

    # Create a new array with the first elements of each consecutive sequence
    filtered_arr = np.delete(arr, consecutive_indices, axis=0)

    return filtered_arr

def check_event_data(events, event_id):
    # Condition: third column equals 1
    condition = events[:, 2] == event_id
    # Create a new array based on the condition
    filtered_array = events[condition]
    
    given_array = filtered_array[:,0]
    # Calculate differences between consecutive values
    differences = np.diff(given_array)
    # Find the indices where the value is less than 30
    indices_less_than_30 = np.where(differences < 30)[0]
    # # Remove rows based on the specified indices
    # selected_array = np.delete(filtered_array, indices_less_than_30, axis=0)
    
    return indices_less_than_30, differences


if __name__ == '__main__':
    seed(123)

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Extract Raw Epochs')

    # Add parameters to the parser
    parser.add_argument('-subject_id', type=str, help='Specify the subject id')
    parser.add_argument('-topic_id', type=str, help='Specify the topic id')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    subject_id = args.subject_id
    topic_id = args.topic_id
    # Dir base
    recordings_dir = 'EEG_data'
    type_dir = 'raw'
    eeg_file = f'article_{topic_id}.vhdr'
    selected_path = Path(recordings_dir) / type_dir / subject_id / eeg_file  # Path to the raw EEG Data folder
    raw = read_raw_data(raw_file_path=selected_path) # Returns a Raw object containing EEG data
    raw_filter = filter_raw(raw, n_jobs = 1)
    # Create raw epochs
    # Convert raw data to BIDS format
    events, events_ids=mne.events_from_annotations(raw_filter, event_id='auto')#read events from raw annotations
    event_id = 1
    tmin = -0.2
    custom_events = customise_events(events)
    indices_less_than_30, differences = check_event_data(custom_events, event_id)
    epochs = create_epochs_from_events(raw=raw_filter, event_ids_list=[event_id])
    drop_indices = [value + 1 for value in indices_less_than_30]
    word_epochs = epochs.copy()
    if len(indices_less_than_30) > 0:
        word_epochs.drop(drop_indices)

    # Save output file
    data_path = 'DERCo'
    topic_folder = 'article_' + str(topic_id)
    # Create a Path object for the file
    epochs_path = Path(data_path) / subject_id / topic_folder

    file_name = 'raw_epoch.fif'

    # Create path to epoch files
    if not os.path.exists(epochs_path):
        os.makedirs(epochs_path)

    raw_epochs = EpochsArray(word_epochs, info=word_epochs.info, tmin=tmin)
    raw_fname = os.path.join(epochs_path, file_name)
    word_epochs.save(raw_fname, overwrite=True)
        

    
from utils.epochs import create_epochs_from_events
from utils.raw import read_raw_data, filter_raw

from pathlib import Path
import numpy as np
import os
import mne
from mne import EpochsArray
import argparse

def customise_events(arr):
    # Find the indices of consecutive elements with the same value in the third column
    consecutive_indices = []
    current_value = arr[0, 2]
    start_index = 0

    for i in range(1, len(arr)):
        if arr[i, 2] != current_value:
            if i - start_index > 1:
                consecutive_indices.extend(range(start_index + 1, i))
            start_index = i
            current_value = arr[i, 2]

    # Check if the last sequence also has consecutive elements
    if len(arr) - start_index > 1:
        consecutive_indices.extend(range(start_index + 1, len(arr)))

    # Create a new array with the first elements of each consecutive sequence
    filtered_arr = np.delete(arr, consecutive_indices, axis=0)

    return filtered_arr

def check_event_data(events, event_id):
    # Condition: third column equals 1
    condition = events[:, 2] == event_id
    # Create a new array based on the condition
    filtered_array = events[condition]
    
    given_array = filtered_array[:,0]
    # Calculate differences between consecutive values
    differences = np.diff(given_array)
    # Find the indices where the value is less than 30
    indices_less_than_30 = np.where(differences < 30)[0]
    # # Remove rows based on the specified indices
    # selected_array = np.delete(filtered_array, indices_less_than_30, axis=0)
    
    return indices_less_than_30, differences


if __name__ == '__main__':

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Extract Raw Epochs')

    # Add parameters to the parser
    parser.add_argument('-subject_id', type=str, help='Specify the subject id')
    parser.add_argument('-topic_id', type=str, help='Specify the topic id')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    subject_id = args.subject_id
    topic_id = args.topic_id
    # Dir base
    recordings_dir = 'EEG_data'
    type_dir = 'raw'
    eeg_file = f'article_{topic_id}.vhdr'
    selected_path = Path(recordings_dir) / type_dir / subject_id / eeg_file  # Path to the raw EEG Data folder
    raw = read_raw_data(raw_file_path=selected_path) # Returns a Raw object containing EEG data
    raw_filter = filter_raw(raw, n_jobs = 1)
    # Create raw epochs
    # Convert raw data to BIDS format
    events, events_ids=mne.events_from_annotations(raw_filter, event_id='auto')#read events from raw annotations
    event_id = 1
    tmin = -0.2
    custom_events = customise_events(events)
    indices_less_than_30, differences = check_event_data(custom_events, event_id)
    epochs = create_epochs_from_events(raw=raw_filter, event_ids_list=[event_id])
    drop_indices = [value + 1 for value in indices_less_than_30]
    word_epochs = epochs.copy()
    if len(indices_less_than_30) > 0:
        word_epochs.drop(drop_indices)

    # Save output file
    data_path = 'DERCo'
    topic_folder = 'article_' + str(topic_id)
    # Create a Path object for the file
    epochs_path = Path(data_path) / subject_id / topic_folder

    file_name = 'raw_epoch.fif'

    # Create path to epoch files
    if not os.path.exists(epochs_path):
        os.makedirs(epochs_path)

    raw_epochs = EpochsArray(word_epochs, info=word_epochs.info, tmin=tmin)
    raw_fname = os.path.join(epochs_path, file_name)
    word_epochs.save(raw_fname, overwrite=True)
        

    
