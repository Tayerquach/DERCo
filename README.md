# DERCo
==============================

The objective of this pipeline is to conduct a preprocessing EEG signals prior to statistical analysis. It reduces the need for manual interventions in data cleaning through visual inspection. It is recommended to review the raw epochs before generating the preprocessed file.
<div class="alert alert-block alert-info">
    <b>Step 1: Import raw data </b> <br>
    EEG raw epoch data will be read via the mne library. <br>
    <b>Step 2: Temporal filtering </b> <br>
    Actually, the raw epochs had been applied this filter
    High-frequency artefacts and slow drifts are removed with a zero-phase bandpass filter using mne-Python. The cutoff frequencies (0.5 - 45 Hz) can be modified in the utils folder in the configuration file (config.py). Using a notch filter at 50Hz to remove AC line current noise. <br>
    <b>Step 3: Get EEG channels and set montage</b> <br>
    HEOG was removed from the raw epochs. The EEG electrodes were positioned via the montage setup.
    <b>Step 4: Re-reference</b> <br>
    We used common average referencing (CAR) to generate a more ideal reference electrode for EEG recordings.
    <b>Step 5: Create metadata</b> <br>
    The <strong>metadata</strong> includes:
    <li><strong>word</strong> (string): the simuli</li>
    <li><strong>NumberOfLetters</strong> (integer): the number of characters in a stimulus</li>
    <li><strong>WordFrequency</strong> (integer): return the word frequency on a human-friendly logarithmic scale.</li>
    <li><strong>WordFrequency</strong> (integer): return the word frequency on a human-friendly logarithmic scale.</li>
    <li><strong>Orthographic Distance & Bigram Frequency</strong> (integer): 
    <div class="alert alert-block alert-info">
    <b>Orthographic distance</b> was defined as the average Levenshtein distance of the 20 most orthographically similar words <a href="https://link.springer.com/article/10.3758/PBR.15.5.971">Yarkoni, Balota, & Yap, 2008</a>, where Levenshtein distance is the minimum number of letter substitutions, deletions, additions, or transpositions required to transform one word into another. <br>
    <b>Bigram Frequency</b> (<a href="https://link.springer.com/article/10.3758/BF03193014">Balota et al,, 2007</a>) refers to summed bigram frequencies by position, and is based on bigram frequencies that are sensitive to positions within words. Position-sensitive bigram counts take into account the letter positions where a bigram occurs. For example, the bigram frequency for DO in DOG counts DO bigrams only when they appear in the first two positions of a word.<br>
    <b>Tool:</b> Using https://elexicon.wustl.edu/
</div>
</li>
<li><strong>ConsonantVowelProportion</strong> (integer): calculated by dividing the number of consonants in a word by the number of letters.</li>
<li><strong>WordID</strong> (string): identifies the position of words in the respective topic.</li>
<li><strong>PredictionRate</strong> (integer): the percentages of prediction based on human performances</li>
<li><strong>Prediction</strong> (boolean): True if the predicted word matched with the correct word and versa.</li>
    <b>Step 6: Run Preprocessing </b> <br>
        <li>Preliminar rejection </li>
            Epochs are rejected based on a global threshold on the z-score (> 3) of the epoch variance and amplitude range.
        <li>Run ICA </li> 
            The default method is the infomax algorithm, however it can be changed in the configuration file along with the number of components and the decimation parameter. Components containing blink artefacts are automatically marked with mne-Python. Then, the mne-ica label library was used to remove the ICs corresponding to artefacts (eye-blink, muscle, heart rate ...).
        <li>Autoreject </li>
            Autoreject [2, 3] uses unsupervised learning to estimate the rejection threshold for the epochs. In order to reduce computation time that increases with the number of segments and channels, autoreject can be fitted on a representative subset of epochs (25% of total epochs). Once the parameters are learned, the solution can be applied to any data that contains channels that were used during fit.<br>
    <b> References </b> <br>
        <li>[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X</li>
        <li>[2] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, “Automated rejection and repair of bad trials in MEG/EEG.” In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.</li>
        <li>[3] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.</li>
        <li>[4] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Frontiers in neuroinformatics, 9, 16.</li>
    
    
</div>

The goal of this project is to generate the preprocessed EEG data in the DERCo dataset. The project has two main stages as follows.
* [Environment Setup](#Environment_Setup)
* [Dataset](#Dataset)
* [Run Preprocessing](#Run_Preprocessing)

Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── article            <- Stimuli.
    │   ├── article_0.pkl      <- File contains the content of the first article.
    │   ├── article_1.pkl      <- File contains the content of the second article.
    │   ├── article_2.pkl      <- File contains the content of the third article.
    │   ├──article_3.pkl       <- File contains the content of the fourth article.
    │   └── article_4.pkl      <- File contains the content of the fifth article
    ├── dataset
    └── Items.csv       <- data generated from The English Lexicon Project.
    ├── human_performance        <- Human Annotation.
    │   ├── article_0_human_performance.csv      <- The results of next-word prediction task from article 0.
    │   ├── article_1_human_performance.csv      <- The results of next-word prediction task from article 1.
    │   ├── article_2_human_performance.csv      <- The results of next-word prediction task from article 2.
    │   ├── article_3_human_performance.csv      <- The results of next-word prediction task from article 3.
    │   └── article_4_human_performance.csv      <- The results of next-word prediction task from article 4.
    ├── utils                   <- contains many functions to preprocess the EEG data.
    │   ├── config.py           <- contains many settings for ICA, filter and processing epochs.
    │   ├── helper.py           <- contains frequently used functions dealing with array.
    │   ├── preprocessing.py    <- contains functions that can be used to clean EEG/MEG data using MNE-Python.
    │   ├── raw.py              <- contains frequently used functions dealing with raw data (creating, filtering, ...).
    │   └── epochs.py           <- contains several functions to generate epochs (from triggers, events, object ...).
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    └── run_preprocessing.py    <- the main function to create the preprocessed epochs from raw epochs.

--------

# Environment Setup
1. Install Python (<a target="_blank" href="https://wiki.python.org/moin/BeginnersGuide">Setup instruction</a>)
2. Install Python packages
```console 
pip3 install -r requirements.txt 
``` 

# Dataset
This preprocessing pipeline was used to generate the DERCo dataset. To download this dataset, please go to this [link](https://osf.io/rkqbu/). For further details about schema and the structure of DERCo dataset, please refer to this [paper]().
# Run Preprocessing
After downloading the DERCo dataset, to create `preprocessed_epoch.fif` for each article per participant, we can run the code below.
```console 
python3 run_preprocessing.py -subject_id=[subject_id] -topic_id=[topic_id]
``` 
For example,
```console 
python3 run_preprocessing.py -subject_id=ACB71 -topic_id=0
```
In this example, the preprocessing pipeline was applied to clean the raw epoch data of the first article for participant ACB71.
