# DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG
# [![Python][Python.py]][Python-url]
## Overview
The objective of this pipeline is to conduct preprocessing on EEG signals for the DERCo dataset prior to statistical analysis. However, this pipeline is also suitable for application in many EEG-based experiments. We only need to change parameters required for our own experiments in the `config.py file`. The most significant function of this pipeline is to reduce the need for manual interventions in data cleaning through visual inspection. For example, when using ICA to decompose the data into a set of components, we must observe them to identify components containing artifacts such as eye blinks, muscle movements, heart rate, etc. These components will then be removed from the data, and the remaining components will be reconstructed into "clean data". However, with this pipeline, this procedure is automatically implemented.

## Description
The next section describes and explains the structure of the preprocessing pipeline. The DERCo dataset will be used in this pipeline. It is recommended to review the raw epochs before generating the preprocessed file. The dataset is available on the Open Science Framework (OSF) at https://osf.io/rkqbu/.

<div class="alert alert-block alert-info">
    <b>Step 1: Import raw data </b> <br>
    EEG raw epoch data will be read via the MNE-Python [1] library. <br>
    <b>Step 2: Temporal filtering </b> <br>
    Actually, the raw epochs had been applied this filter
    High-frequency artefacts and slow drifts are removed with a zero-phase bandpass filter using mne-Python. The cutoff frequencies (0.1 - 45 Hz) can be modified in the utils folder in the configuration file (<em>config.py</em>). Using a notch filter at 50Hz to remove AC line current noise. <br>
    <b>Step 3: Get EEG channels and set montage</b> <br>
    HEOG was removed from the raw epochs. The EEG electrodes were positioned via the montage setup. <br>
    <b>Step 4: Re-reference</b> <br>
    We used common average referencing (CAR) to generate a more ideal reference electrode for EEG recordings. <br>
    <b>Step 5: Create metadata</b> <br>
    Please refer to https://osf.io/rkqbu/wiki/Schema/ to get more information. <br>
    <b>Step 6: Run Preprocessing </b> <br>
        <li>Preliminar rejection </li>
            Epochs are rejected based on a global threshold on the z-score (> 3) of the epoch variance and amplitude range.
        <li>Run ICA </li> 
            The default method is the infomax algorithm, however it can be changed in the configuration file along with the number of components and the decimation parameter. Components containing blink artefacts are automatically marked with mne-Python. Then, the mne-ica label library was used to remove the ICs corresponding to artefacts (eye-blink, muscle, heart rate ...).
        <li>Autoreject </li>
            Autoreject [2, 3] uses unsupervised learning to estimate the rejection threshold for the epochs. In order to reduce computation time that increases with the number of segments and channels, autoreject can be fitted on a representative subset of epochs (25% of total epochs). Once the parameters are learned, the solution can be applied to any data that contains channels that were used during fit.<br>
    
    
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

## Environment Setup
After cloning repository github, going to the DERCo folder and do the steps as follows

1. If you have not installed Python, please install it first:

   Install Python (<a target="_blank" href="https://wiki.python.org/moin/BeginnersGuide">Setup instruction</a>).
   
   **Note**: In this project, we used Python 3.10.9
3. Install Python packages
```console 
pip3 install -r requirements.txt 
``` 

## Dataset
This preprocessing pipeline was used to generate the DERCo dataset. To download this dataset, please go to this [link](https://osf.io/rkqbu/). Download `EEG_data` folder and put it into the DERCo directory with the same level as `article`. All **raw data** from each subject was stored in `raw` folder. We can download `raw_epoch.fif` files only and then run this pipeline. After that, the results can are validated by the corresponding available `preprocessed_epoch.fif` files in `preprocessed` folders.
For further details about schema and the structure of DERCo dataset, please refer to this [paper]().
## Run Preprocessing
After downloading the DERCo dataset, to create `preprocessed_epoch.fif` for each article per participant, we can run the code below.
```console 
python3 run_preprocessing.py -subject_id=[subject_id] -topic_id=[topic_id]
``` 
For example,
```console 
python3 run_preprocessing.py -subject_id=ACB71 -topic_id=0
```
In this example, the preprocessing pipeline was applied to clean the raw epoch data of the first article for participant ACB71.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References
[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X.

[2] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, “Automated rejection and repair of bad trials in MEG/EEG.” In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[3] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.

Meeg-tools: https://github.com/weiglszonja/meeg-tools

<!-- MARKDOWN LINKS & IMAGES -->
[Python.py]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
