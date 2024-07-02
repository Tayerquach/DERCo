from random import seed
import os
import mne
from mne import EpochsArray
from utils.preprocessing import *

import numpy as np
import pandas as pd
import pickle
import argparse
from mne_icalabel import label_components


from wordfreq import zipf_frequency

import warnings
warnings.filterwarnings('ignore')


def count_letters(input_string):
    letter_count = 0
    for char in input_string:
        if char.isalpha():
            letter_count += 1
    return letter_count
def count_consonants(word):
    consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
    count = 0
    for char in word:
        if char in consonants:
            count += 1
    return count

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
    # Import raw data
    data_path = 'EEG_data'
    input_folder = 'raw'
    topic_folder = 'article_' + str(topic_id)
    file_name = 'raw_epoch.fif'
    subject_input_folder = Path(data_path) / input_folder / subject_id / topic_folder 
    raw_fname = os.path.join(subject_input_folder, file_name)
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
    custom_epochs = raw_epochs.copy().set_eeg_reference("average", projection=True).apply_proj() # re-referencing with the virtual average reference
    times = np.arange(custom_epochs.tmin, custom_epochs.tmax, 0.1) #[min],[max],[stepsize]

    '''
        Create metadata
    '''
    #1. Word
    with open(f'article/article_{topic_id}.pkl', 'rb') as f:
        word_stimulus = pickle.load(f)
    stimuli = [w + f'_{topic_id}_{i}'for i, w in enumerate(word_stimulus)]
    metadata = pd.DataFrame({
        'WORD': word_stimulus,
        'word': stimuli
    })
    #2. NumberOfLetters
    metadata['NumberOfLetters'] = metadata['WORD'].apply(lambda x: count_letters(x))
    #3. WordFrequency
    metadata['WordFrequency'] = metadata['WORD'].apply(lambda x: zipf_frequency(x, 'en'))
    #4. Orthographic Distance & Bigram Frequency
    stat_df = pd.read_csv('dataset/Items.csv')
    temp = stat_df[['Word', 'OLD', 'BG_Freq_By_Pos']]
    temp.rename(columns={'Word': 'WORD', 'OLD':'OrthographicDistance', 'BG_Freq_By_Pos':'BigramFrequency'}, inplace=True)
    metadata = pd.merge(metadata, temp, on='WORD', how='left')
    #5. ConsonantVowelProportion
    metadata['ConsonantVowelProportion'] = metadata['WORD'].apply(lambda x: count_consonants(x)/count_letters(x))
    #6. Prediction
    #Load human_performance
    human_perf = pd.read_csv(f'human_performance/article_{topic_id}_human_performance.csv')

    metadata['HumanResponse'] = ['', ''] + list(human_perf['response'].values)
    metadata['WordID'] = ['', ''] + list(human_perf['word_id'].values)
    metadata['PredictionRate'] = [np.nan, np.nan] + list(human_perf['percentage'].values)
    metadata['Prediction'] = metadata['WORD'] == metadata['HumanResponse']
    metadata = metadata.drop(columns=['HumanResponse'])

    # Replace '#' with NaN in BigramFrequency
    metadata['BigramFrequency'] = metadata['BigramFrequency'].replace('#', np.nan)
    # Replace the first two values with NaN in Prediction
    metadata['Prediction'].iloc[:2] = np.nan
    # Reorder columns
    column_order = ['WordID', 'word', 'NumberOfLetters', 'WordFrequency', 'OrthographicDistance', 'BigramFrequency', 'ConsonantVowelProportion', 'Prediction', 'PredictionRate']
    metadata = metadata[column_order]
    # Reset the index and move it to a regular column
    metadata.reset_index(inplace=True)
    custom_epochs.metadata = metadata

    '''
        Run Preprocessing
    '''
    seed(123)
    #Preliminary epoch rejection
    epochs_faster, bad_epochs = prepare_epochs_for_ica(epochs=custom_epochs)

    #Run ICA
    ica = run_ica(epochs=epochs_faster)
    ica_data = ica.get_sources(epochs_faster).get_data()
    ic_labels = label_components(epochs_faster, ica, method="iclabel")
    labels = ic_labels["labels"]
    excluded_ics = [i for i, artifact in enumerate(labels) if artifact not in ["brain", "other"]]
    name_excls = [artifact for artifact in labels if artifact not in ["brain", "other"]]
    signal_names = list(set(name_excls))
    brain_ics = [i for i, artifact in enumerate(labels) if artifact in ["brain", "other"]]
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = epochs_faster.copy()
    ica.apply(reconst_raw.load_data(), exclude=excluded_ics)
    channels_id = [i for i in range(len(channel_names))]

    #Run autoreject
    reject_log = run_autoreject(reconst_raw.load_data(), n_jobs=11, subset=True)
    # rejecting only bad epochs
    epochs_autoreject = reconst_raw.copy().drop(reject_log.report, reason='AUTOREJECT')
    preprocessed_epochs = epochs_autoreject.copy()

    #Save preprocessed epochs
    epochs_data = preprocessed_epochs['1'].get_data()

    #Customize event data
    cleaned_words = preprocessed_epochs.metadata['word'].values
    # stimuli = [w + f'_{topic_id}_{i}'for i, w in enumerate(cleaned_words)]
    stimuli = cleaned_words.copy()
    indice = preprocessed_epochs.metadata['index'].values
    event_id_new = dict(zip(stimuli, indice))
    column_1 = [epochs_data.shape[2]*i for i in range(len(cleaned_words))]
    # column_2 = column_3 = [i for i in range(len(cleaned_words))]
    column_2 = column_3 = indice.copy()
    # Create a 3D NumPy array
    events_data = np.array([column_1, column_2, column_3])
    events_data_new = events_data.T

    # create preprocessed epochs object 
    tmin = -0.2
    new_cleaned_metadata = preprocessed_epochs.metadata.drop('index', axis=1)
    preprocessed_epochs = EpochsArray(epochs_data, info=preprocessed_epochs.info, events=events_data_new.astype('int'), event_id=event_id_new, tmin=tmin, metadata = new_cleaned_metadata)

    # Save output file
    data_path = 'EEG_data'
    output_folder = 'preprocessed'
    topic_folder = 'article_' + str(topic_id)
    file_cleaned_name = 'preprocessed_epoch.fif'
    subject_output_folder = Path(data_path) / output_folder / subject_id / topic_folder
    # Check if the folder exists, if not create it
    if not os.path.exists(subject_output_folder):
        os.makedirs(subject_output_folder)

    cleaned_fname = os.path.join(subject_output_folder, file_cleaned_name)
    preprocessed_epochs.save(cleaned_fname, overwrite=True)

    

