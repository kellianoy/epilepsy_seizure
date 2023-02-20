import os
import numpy as np
from mne.io import read_raw_edf
import csv
import urllib.request
import shutil
import warnings
# This is a script to download the data from CHB-MIT scalp EEG database
# The dataset can be found here: https://archive.physionet.org/pn6/chbmit/
# Or here: https://physionet.org/content/chbmit/1.0.0/
# Recordings, grouped into 23 cases, were collected from 22 subjects


def data_to_fft(data):
    """
        Returns n x T array, n and T being respectively number of channel in data,
        and number of time bin in data

        Parameters
        ----------
        data:  array of data from CHB-MIT dataset

        Returns
        ----------
        numpy array
    """

    fft_tensor = np.fft.rfft(data[:, 1:], axis=0)
    fft_tensor = np.float16(np.log10(np.abs(fft_tensor)+1e-6))
    indices = np.where(fft_tensor <= 0)
    fft_tensor[indices] = 0

    return fft_tensor  # ,freq_array,time_array


def edf_to_array(filename_in, seizures_time_code, time_length, number_of_patient):
    """ Convert an edf file to a numpy array
    Parameters
    ----------
    filename_in: str
        Name of the file to convert
    seizures_time_code: list
        List of the time of the seizures
    time_length: float
        Length of the time window in seconds
    number_of_patient: int

    Returns
    ----------
    x: numpy array
        Array of the data
    y: numpy array
        Array of the labels
    """

    if number_of_patient == 16:
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1',
               u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
    else:
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1',
               u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']

    rawEEG = read_raw_edf('%s' % (filename_in),
                          # exclude=exclude_chs,  #only work in mne 0.16
                          verbose=0, preload=True)

    rawEEG.pick_channels(chs)
    tmp = rawEEG.to_data_frame()
    tmp = tmp.to_numpy()
    time_array = tmp[:, 0]
    freq_mean = 1000/np.mean(tmp[1:, 0]-tmp[:-1, 0])

    time_iterator = tmp[0, 0]
    x, y = [], []
    while time_iterator*1000 + time_length*1000 < tmp[-1, 0]:
        index_start = int(time_iterator*freq_mean)
        index_stop = int(index_start + time_length*freq_mean)
        data = tmp[index_start:index_stop, 1:]

        flag_ictal = 0

        for bounds in seizures_time_code:
            if (bounds[0] < tmp[index_start, 0]/1000 < bounds[1]) and (bounds[0] < tmp[index_stop, 0]/1000 < bounds[1]):
                flag_ictal = 1

        x.append(data)
        y.append([flag_ictal, tmp[index_start, 0]])
        time_iterator += time_length*(1-flag_ictal) + flag_ictal*2/(256)

    return np.array(x), np.array(y)


def preprocess_to_numpy(records_path, seizure_summary_path, database_path, number_of_patient, dir_where_to_save, time_length):
    """ Preprocess the data from the CHB-MIT dataset and save it in numpy format
    Parameters
    ----------
    records_path: str
        Path to the file containing the `RECORDS` file
    seizure_summary_path: str
        Name of the file containing the list of the filename of the data
    database_path: str
        Path to the folder containing the data
    number_of_patient: int
        Number of the patient
    dir_where_to_save: str
        Path to the folder where to save the data
    time_length: float
        Length of the time window in seconds
    """
    # Ignore warnings
    warnings.filterwarnings("ignore")
    csv_file = open(seizure_summary_path)
    csv_reader_bounds = csv.reader(csv_file, delimiter=',')
    liste_bounds = [[], [], []]
    for row in csv_reader_bounds:
        if row != [] and row != ['File_name', 'Seizure_start', 'Seizure_stop']:
            liste_bounds[0].append(row[0])
            liste_bounds[1].append(float(row[1]))
            liste_bounds[2].append(float(row[2]))

    csv_file = open(records_path)
    csv_reader_list_filename = csv.reader(csv_file, delimiter=',')

    flag = False
    for filename in csv_reader_list_filename:
        if int(filename[0][3]+filename[0][4]) == number_of_patient:
            bounds = []
            if filename[0][6:] in liste_bounds[0]:
                indices = [i for i, x in enumerate(
                    liste_bounds[0]) if x == filename[0][6:]]
                for indice in indices:
                    bounds.append([liste_bounds[1][indice],
                                  liste_bounds[2][indice]])
            x, y = edf_to_array(database_path +
                                filename[0], bounds, time_length, number_of_patient)
            if not flag:
                x_master = np.copy(x)
                y_master = np.copy(y)
                flag = True
            else:
                x_master = np.concatenate((x_master, x))
                y_master = np.concatenate((y_master, y))

    patient_file = f"chb0{number_of_patient}"if number_of_patient < 10 else f"chb{number_of_patient}"
    if not os.path.exists(dir_where_to_save + patient_file):
        os.makedirs(dir_where_to_save + patient_file)
    np.save(dir_where_to_save + patient_file + "/" + patient_file +
            "_X.npy", np.float16(x_master))
    np.save(dir_where_to_save + patient_file + "/" +
            patient_file + "_y.npy", np.float16(y_master))


def download_dataset(eeg_database_folder, remove=False, force_process=False, force_download=False):
    """ Download the dataset from the website, limiting the number of records
    (some records are not working)
    Parameters
    ----------
    eeg_database_folder: str
        eeg_database_folder where to save the dataset
    remove: bool
        if True, remove the eeg_database_folder before downloading the dataset to save space
    force_process: bool
        if True, force the processing of the dataset
    force_download: bool
        if True, force the download of the dataset
    """
    # file where to save the records file (header)
    seizure_summary = eeg_database_folder + "seizure_summary.csv"
    records_summary = eeg_database_folder + "RECORDS"
    dataset_folder = 'dataset/'
    # website where to download the dataset
    website = "https://archive.physionet.org/pn6/chbmit/"
    summary = "https://raw.githubusercontent.com/NeuroSyd/Integer-Net/master/copy-to-CHBMIT/seizure_summary.csv"
    # download the records file (header)
    if not os.path.exists(eeg_database_folder):
        os.makedirs(eeg_database_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(records_summary):
        urllib.request.urlretrieve(website+"RECORDS", records_summary)
    # download the seizure summary file
    if not os.path.exists(seizure_summary):
        urllib.request.urlretrieve(summary, seizure_summary)
    # For each patient we are interested in, download the records
    patients = {}
    for i in [2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23, 24]:
        # Retrieve summary eeg_database_folderof seizures
        current_patient = f"chb0{i}" if i < 10 else f"chb{i}"
        patients[current_patient] = i

    # Open records summary, and for each line, download the record.
    previous_patient = None
    with open(records_summary) as f:
        for record in f:
            # patients is a dictionary of patients we are interested in
            if record[:5] in patients.keys():
                patient = record[:5]
                if not os.path.exists(eeg_database_folder + patient):
                    os.makedirs(eeg_database_folder + patient)
                # Preprocess the data and save it in numpy format
                if patient != previous_patient and previous_patient is not None:
                    # Preprocess the previous patient
                    if not os.path.exists(dataset_folder + previous_patient) or force_process:
                        print("Preprocessing " + previous_patient + "...")
                        preprocess_to_numpy(records_summary,
                                            seizure_summary,
                                            eeg_database_folder,
                                            patients[previous_patient],
                                            dataset_folder,
                                            1)
                    if remove:
                        # Remove the previous patient eeg_database_folder to save space
                        print("Removing " + previous_patient + "...")
                        shutil.rmtree(eeg_database_folder+previous_patient)
                # Download the record if it does not exist
                if not os.path.exists(eeg_database_folder+record.strip()) or force_download:
                    print("Downloading "+record.strip() + "...")
                    urllib.request.urlretrieve(
                        website+record.strip(), eeg_database_folder+record.strip())
                previous_patient = patient
    if remove:
        print("Removing " + eeg_database_folder + "...")
        shutil.rmtree(eeg_database_folder)


if __name__ == "__main__":

    folder = "chb-mit-scalp-eeg-database-1.0.0/"
    download_dataset(folder)
