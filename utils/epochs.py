from typing import List
import numpy as np
import pandas as pd
import os
import mne 
import collections

from mne import (
    Epochs,
    find_events,
    events_from_annotations,
    make_fixed_length_events,
    merge_events,
    concatenate_epochs,
    pick_events,
    create_info,
    EpochsArray
)
from mne.epochs import combine_event_ids
from mne.io import Raw
from mne.utils import logger

from .config import settings
from utils.helper import blockshaped

def create_epochs_from_events(raw: Raw, event_ids_list: List) -> Epochs:
    """
    Create non-overlapping segments from Raw data.
    Note that temporal filtering should be done before creating the epochs.
    If there are annotations (triggers) found in the raw data, it creates
    epochs with respect to the stimulus onset defined by start_time
    and end_time in the configuration file (config.py) in seconds.
    Parameters
    ----------
    raw: the continuous data to be segmented into non-overlapping epochs
    event_ids: the list of event ids to create epochs from


    Returns
    -------
    Epochs instance
    """


    try:
        events_data = find_events(raw)
    except ValueError:
        events_data, _ = events_from_annotations(raw)

    # event_ids_list = list(events_ids.values())
    selected_events = pick_events(events_data, include=event_ids_list)#pick events  that we interested
    # selected_events = events[np.isin(events[..., 2], event_ids)]
    logger.info("Creating epochs from selected events ...")
    epochs = Epochs(
        raw=raw,
        events=selected_events,
        picks="all",
        event_id=event_ids_list,
        baseline=None,
        tmin=settings["epochs"]["start_time"],
        tmax=settings["epochs"]["end_time"],
        preload=True,
        # reject=dict(eeg=500e-6) #reject those larger than 300 microvolt
    )

    return epochs

def create_epochs(raw: Raw) -> Epochs:
    """
    Create non-overlapping segments from Raw data with a fixed duration.
    Note that temporal filtering should be done before creating the epochs.
    The duration of epochs is defined in the configuration file (config.py).
    Parameters
    ----------
    raw: the continuous data to be segmented into non-overlapping epochs
    Returns
    -------
    Epochs instance
    """

    epoch_duration_in_seconds = settings["epochs"]["duration"]
    logger.info("Creating epochs from continuous data ...")
    events = make_fixed_length_events(
        raw, id=1, first_samp=True, duration=epoch_duration_in_seconds
    )

    epochs = Epochs(
        raw=raw,
        events=events,
        picks="all",
        event_id=list(np.unique(events[..., 2])),
        baseline=None,
        tmin=0.0,
        tmax=epoch_duration_in_seconds,  # - (1 / raw.info['sfreq']
        preload=False,
    )

    return epochs   

def create_epochs_from_intervals(raw: Raw, intervals: List[tuple]) -> Epochs:
    events, _ = events_from_annotations(raw)

    epochs_list = []
    for interval in intervals:
        start_idx = np.where(events[..., 2] == interval[0])[0]
        end_idx = np.where(events[..., 2] == interval[1])[0]

        raw_cropped = raw.copy().crop(
            tmin=events[start_idx[0]][0] / raw.info["sfreq"],
            tmax=events[end_idx[0]][0] / raw.info["sfreq"],
        )

        epochs = create_epochs(raw_cropped)
        combine_event_ids(epochs, list(epochs.event_id.keys()), interval[0], copy=False)

        epochs_list.append(epochs)

    return concatenate_epochs(epochs_list)

def extract_tracker_log(file_logs: str):
    """
    Extract file log (.pkl) and convert it into a dataframe
    ----------
    Parameters
    file_logs: the full path to the log file
    -------
    Returns
    data: Tracker dataframe
    """
    
    word_data  = pd.read_pickle(file_logs)
    data       = pd.DataFrame()
    
    total_session = len(word_data)
    for session_num in range(total_session):
        epoch_data    = pd.DataFrame.from_dict(word_data[session_num])
        session_data  = epoch_data['session'].apply(pd.Series)

        #Parse trial infomation
        trial_data = pd.DataFrame()
        for i in range(len(session_data['trial_info'])):
            temp = pd.DataFrame.from_dict(session_data['trial_info'][i])
            trial_data = pd.concat([temp, trial_data], ignore_index=True)
        trial_data = trial_data.sort_values('index_trials', ignore_index=True)

        #Parse event data
        event_df = pd.DataFrame()
        for trial in range(len(trial_data['events'])):
            values = []
            db = pd.DataFrame.from_dict(trial_data['events'][trial])
            values = db['event_time'].values[0]
            arr = values.copy()
            arr.insert(2, arr[2]) # Missed trial time in code
            cols = db['event_name'].values[0]
            temp = pd.DataFrame([arr], columns=cols)
            event_df = pd.concat([temp, event_df], ignore_index=True)
        event_df = event_df.sort_values('start_fixation_cross', ignore_index=True)

        #Parse latency
        latency_df = pd.DataFrame()
        for trial in range(len(trial_data['events'])):
            values = []
            db = pd.DataFrame.from_dict(trial_data['latency'][trial])
            values = db['latency_s'].values[0]
            cols = db['latency_event'].values[0]
            temp = pd.DataFrame([values], columns=cols)
            latency_df = pd.concat([temp, latency_df], ignore_index=True)
        latency_df = latency_df.rename(columns={'stimulus': 'stimulus_duration'}) #Next time, changing in experiment code.
        latency_df = latency_df.sort_values('trial', ignore_index=True)

        #Join dataframe
        ##1. Join epoch and session
        df1 = (epoch_data.drop(['session'], axis = 1).join(session_data).reset_index(drop=True))
        ##2. Join epoch_session and trial
        df2 = (df1.drop(['trial_info'], axis = 1).join(trial_data).reset_index(drop=True))
        ##3. Join with events
        df3 = (df2.drop(['events'], axis = 1).join(event_df).reset_index(drop=True))
        ##4. Join with latency
        df4 = (df3.drop(['latency'], axis = 1).join(latency_df).reset_index(drop=True))
    
        data = pd.concat([df4, data], ignore_index=True)
        data = data.sort_values(by=['index_block', 'index_session', 'index_trials'], ascending=True, ignore_index = True)
    
    return data


def create_epochs_dataframe_with_trigger(tracker_data, epochs):
    """
    Convert epochs into a dataframe and calibrate stimulus based on triggers. 
    ----------
    Parameters
    tracker_data: data contains stimulus' information displayed during experiments.
    epochs: epochs data extracted from EEG file
    -------
    Returns
    eeg_data_new: A new dataframe of epochs.
    """
    
    #Get metadata from corpus 
    kiloword_data_folder = ('dataset/kilo-word-dataset/MNE-kiloword-data')
    kiloword_data_file = os.path.join(kiloword_data_folder, 'kword_metadata-epo.fif')
    epochs_label = mne.read_epochs(kiloword_data_file)
    
    #Merge our tracker data and labels
    merged_data = tracker_data[['stimulus']].merge(epochs_label.metadata, how = 'left', left_on = ['stimulus'], right_on = ['WORD'])
    
    #Find real words
    real_words_data = merged_data.dropna(thresh=2)
    real_words_data = real_words_data.drop(columns=['WORD'])
    
    #Mapping metadata orders into epochs' data
    epochs_dfx = epochs.to_data_frame()
    eeg_data = epochs_dfx[epochs_dfx['condition'] == 'start_stimulus']
    
    #Mapping events (start_stimulus)
    start_stimulus_events = epochs['start_stimulus'].events.shape[0]
    mapping_events = dict(zip(eeg_data['epoch'].unique(), list(range(0,start_stimulus_events))))
    eeg_data = eeg_data.replace({"epoch":mapping_events}) # 550 start_stimulus events
    
    #Mapping stimulus
    eeg_data = eeg_data[eeg_data['epoch']%2 == 0] # 2 start_stimulus events create 1 stimuli
    mapping_stimulus = dict(zip(eeg_data['epoch'].unique(), list(range(0,int(start_stimulus_events/2)))))
    eeg_data = eeg_data.replace({"epoch":mapping_stimulus})
    
    #Mapping condition
    mapping_labels = dict(zip(eeg_data['epoch'].unique(), merged_data['stimulus'].values))
    eeg_data['labels'] = eeg_data['epoch'].values
    eeg_data = eeg_data.replace({"labels":mapping_labels})
    eeg_data = eeg_data.drop(columns=['condition'])
    eeg_data = eeg_data.rename(columns={'labels': 'condition'})
    
    #Remove pseudo-word
    eeg_data_real_word = eeg_data[eeg_data['condition'].isin(real_words_data['stimulus'].values)]
    
    #Check duplicates in tracker logs
    duplicates = [item for item, count in collections.Counter(tracker_data['stimulus'].values).items() if count > 1]
    time_points = int(eeg_data_real_word.shape[0]/real_words_data.shape[0]) #1151 points (0.15s - 1s)
    if len(duplicates) > 0:
        print("Warning: There are duplicates in the data")
        #Drop duplicates but keep the given amount
        eeg_data_duplicates = pd.DataFrame()
        temp = pd.DataFrame()
        for word in duplicates:
            temp = eeg_data_real_word[eeg_data_real_word['condition'] == word]
            temp = temp.iloc[:time_points]
            eeg_data_duplicates = pd.concat([eeg_data_duplicates, temp], ignore_index=True)
            
        temp = eeg_data_real_word[~eeg_data_real_word['condition'].isin(duplicates)]
        eeg_data_new = pd.concat([eeg_data_duplicates, temp], ignore_index=True)
        eeg_data_new = eeg_data_new.sort_values(['epoch', 'time'], ignore_index=True)
        eeg_data_new['labels'] = eeg_data_new['condition'].values

        mapping_epochs_id = dict(zip(eeg_data_new['condition'].unique(), list(range(0,int(eeg_data_real_word.shape[0]/time_points)))))

        eeg_data_new = eeg_data_new.replace({"labels":mapping_epochs_id})
        eeg_data_new = eeg_data_new.drop(columns=['epoch'])
        eeg_data_new = eeg_data_new.rename(columns={'labels': 'epoch'})
        
    else:
        eeg_data_new = eeg_data_real_word.sort_values(['epoch', 'time'], ignore_index=True)
        eeg_data_new['labels'] = eeg_data_new['condition'].values

        mapping_epochs_id = dict(zip(eeg_data_new['condition'].unique(), list(range(0,int(eeg_data_real_word.shape[0]/time_points)))))

        eeg_data_new = eeg_data_new.replace({"labels":mapping_epochs_id})
        eeg_data_new = eeg_data_new.drop(columns=['epoch'])
        eeg_data_new = eeg_data_new.rename(columns={'labels': 'epoch'})

    columns = eeg_data_new.columns.values
    cols_labels = [columns[0]] + list(columns[-2:])
    cols_data = list(columns[1:-2])
    new_columns = cols_labels + cols_data

    eeg_data_new = eeg_data_new[new_columns]
        
    return eeg_data_new, real_words_data

    
def create_epoch_object(epoch_dataframe, metadata, ch_names, ch_type, sfreq, montage_type):

    """
    Convert epoch dataframe into a object included metadata 
    ----------
    Parameters
    epoch_dataframe: A dataframe of given epochs.
    metadata: The information of stimulus (condition)
    ch_names: Channel names (list of str or int). If an int, a list of channel names will be created from range(ch_names).
    ch_type: Channel types (list of str or int). Example, "eeg", "ecg", "eog" .... If str, then all channels are assumed to be of the same type.
    sfreq: Sample rate of the data (float).
    montage_type: The name of the montage to use. See https://mne.tools/dev/generated/mne.channels.make_standard_montage.html for more information.
    -------
    Returns
    custom_epochs: An epoch object.
    """

    # Create epoch's information
    info_epoch = create_info(ch_names, sfreq, ch_type, None)
    info_epoch.set_montage(montage_type)
    info_epoch['description'] = 'Custom dataset'

    #Customize Array
    arr_list = []
    all_epochs_data = epoch_dataframe.values[:,3:]
    num_epochs      = len(epoch_dataframe['condition'].unique())
    num_time_points = len(epoch_dataframe['time'].unique())
    num_channels    = len(ch_names)
    for epoch_id in range(num_epochs):
        epoch_data = blockshaped(all_epochs_data, num_time_points, num_channels)[epoch_id].T
        arr_list.append(epoch_data)
        temp_data = np.vstack(arr_list)
    EEGdata = temp_data.reshape(num_epochs,num_channels,num_time_points)
    EEGdata = EEGdata*10e-7
    EEGdata.shape

    #Customize event data, event id
    temp            = epoch_dataframe[epoch_dataframe.index%1151 == 0]
    temp['time']    = temp.index.values
    events_id_new   = dict(zip(temp['condition'].values, temp['epoch'].values))
    temp            = temp.replace({"condition":events_id_new})
    events_data_new = temp.values[:,:3]

    # create raw object 
    tmin = settings["epochs"]["start_time"]
    custom_epochs = EpochsArray(EEGdata, info=info_epoch, events=events_data_new.astype('int'), tmin=tmin, event_id=events_id_new, metadata = metadata)

    return custom_epochs

