import os
import argparse

import mne
from mne import viz, create_info, EpochsArray, Epochs
from utils.raw import read_raw_data, filter_raw, convert_data_to_BIDS
from utils.epochs import create_epochs_from_events, extract_tracker_log, create_epochs_dataframe_with_trigger, create_epoch_object
from utils.preprocessing import *

import numpy as np
import pandas as pd
from random import seed
import pickle
import collections
from mne_icalabel import label_components
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
from nltk.corpus import brown

from wordfreq import zipf_frequency
from old20 import old20, old_n

import warnings
warnings.filterwarnings('ignore')

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

    '''
        Import raw epochs
    '''

    # Save output file
    data_path = 'DERCo'
    topic_folder = 'article_' + str(topic_id)
    file_name = 'raw_epoch.fif'
    raw_fname = Path(data_path) / subject_id / topic_folder / file_name # Path to the raw epoch folder
    raw_epochs = mne.read_epochs(raw_fname)

    '''
        Get EEG channels and set montage
    '''
    # Remove EOG channels
    raw_epochs.drop_channels(['HEOG'])
    # raw_data.drop_channels(['HEOG', 'VEOG'])
    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020') # Electrode position file
    raw_epochs.set_montage(montage, verbose=False, on_missing='ignore')
    channel_names = raw_epochs.ch_names
    raw_epochs.info

    '''
        Re-reference
    '''
    tmix = -0.2
    tmax = 0.5
    custom_epochs = raw_epochs.copy().set_eeg_reference("average", projection=True).apply_proj() # re-referencing with the virtual average reference
    times = np.arange(custom_epochs.tmin, custom_epochs.tmax, 0.1) #[min],[max],[stepsize]

