import os
import numpy as np
from mne.io import read_raw_edf
import csv
import urllib.request
import urllib.error
import warnings
from multiprocessing import cpu_count, Pool


# This is a script to download the data from CHB-MIT scalp EEG database
# The dataset can be found here: https://archive.physionet.org/pn6/chbmit/
# Or here: https://physionet.org/content/chbmit/1.0.0/
# Recordings, grouped into 23 cases, were collected from 22 subjects
PATIENTS_LIST = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23, 24]
PATIENTS_SIZE = len(PATIENTS_LIST)


def edf_to_array(filename_in, seizures_time_code, time_length, number_of_patient):
    """ Convert an edf file containing eeg of seizures to a numpy array
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
    channel_names = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1',
                     u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
    if number_of_patient != 16:
        channel_names.remove(u'FZ-CZ')
    # Only works in MNE 1.0.0
    raw_egg = read_raw_edf('%s' % (filename_in), verbose=0, preload=True)
    raw_egg.pick_channels(channel_names)
    tmp = raw_egg.to_data_frame().to_numpy()
    freq_mean = 1000 / np.mean(tmp[1:, 0] - tmp[:-1, 0])
    time_iterator = tmp[0, 0]
    X, y = [], []
    while (time_iterator + time_length)*1000 < tmp[-1, 0]:
        index_start = int(time_iterator * freq_mean)
        index_stop = int(index_start + time_length * freq_mean)
        data = tmp[index_start:index_stop, 1:]
        flag_ictal = 0
        for bounds in seizures_time_code:
            if (bounds[0] < tmp[index_start, 0] / 1000 < bounds[1]) and (
                    bounds[0] < tmp[index_stop, 0] / 1000 < bounds[1]):
                flag_ictal = 1
        X.append(data)
        y.append([flag_ictal, tmp[index_start, 0]])
        time_iterator += time_length * (1 - flag_ictal) + flag_ictal * 2 / 256
    return np.array(X), np.array(y)


def remove_hours(X, y):
    """ Remove 4 hours before and after an epilepsy seizure
    Parameters
    ----------
    X: numpy array
        Array of the data
    y: numpy array
        Array of the labels

    Returns
    ----------
    X_interictal: numpy array
        Array of the data with 4 hours before and after an epilepsy seizure removed
    y_interictal: numpy array
        Array of the labels with 4 hours before and after an epilepsy seizure removed
    """
    acc = 0
    y[:, 1] = y[:, 1]/1000
    y_new = np.zeros(y.shape[0])
    for i in range(y.shape[0]-1):
        y_new[i] = acc + y[i, 1]
        if y[i+1, 1] == 0.0:
            # print(i,y[i-1,1])
            acc += y[i, 1]
    y_new[-1] = acc + y[-1, 1]
    y_interictal = np.ones(y_new.shape)
    ictal_memory = 0
    flag_bord = 0
    for i in range(len(y[:, 1])-1):
        if y[i, 0] == 1 and y[i+1, 0] == 0:
            ictal_memory = i
            flag_bord = 1
        if (y_new[i] < y_new[ictal_memory]+4*3600 or y[i, 0] == 1) and flag_bord:
            y_interictal[i] = 0
    ictal_memory = -1
    flag_bord = 0
    for i in range(len(y[:, 1])-1, 0, -1):
        if y[i, 0] == 1 and y[i-1, 0] == 0:
            ictal_memory = i
            flag_bord = 1
        if (y_new[i] > y_new[ictal_memory]-4*3600 or y[i, 0] == 1) and flag_bord:
            y_interictal[i] = 0
    x_to_save = []
    y_to_save = []
    for i in range(y.shape[0]):
        if y[i, 0] == 1:
            x_to_save.append(X[i])
            y_to_save.append(1)
        elif y_interictal[i] == 1:
            x_to_save.append(X[i])
            y_to_save.append(0)
    return np.array(x_to_save), np.array(y_to_save)


def preprocess_to_numpy(records_path, seizure_summary_path, database_path, patient_id, output_folder,
                        time_length):
    """ Preprocess the data from the CHB-MIT dataset and save it in numpy format
    Parameters
    ----------
    records_path: str
        Path to the file containing the `RECORDS` file
    seizure_summary_path: str
        Name of the file containing the list of the filename of the data
    database_path: str
        Path to the folder containing the data
    patient_id: int
        Number of the patient
    output_folder: str
        Path to the folder where to save the data
    time_length: float
        Length of the time window in seconds
    """
    # Ignore warnings
    warnings.filterwarnings("ignore")
    csv_reader_bounds = csv.reader(
        open(seizure_summary_path, 'r'), delimiter=',')
    liste_bounds = [[], [], []]
    for row in csv_reader_bounds:
        if row != [] and row != ['File_name', 'Seizure_start', 'Seizure_stop']:
            liste_bounds[0].append(row[0])
            liste_bounds[1].append(float(row[1]))
            liste_bounds[2].append(float(row[2]))
    csv_reader_list_filename = csv.reader(open(records_path), delimiter=',')
    flag = False
    for filename in csv_reader_list_filename:
        if int(filename[0][3] + filename[0][4]) == patient_id:
            bounds = []
            if filename[0][6:] in liste_bounds[0]:
                indices = [i for i, x in enumerate(
                    liste_bounds[0]) if x == filename[0][6:]]
                for indice in indices:
                    bounds.append([liste_bounds[1][indice],
                                   liste_bounds[2][indice]])
            x, y = edf_to_array(database_path +
                                filename[0], bounds, time_length, patient_id)
            if not flag:
                x_master = np.copy(x)
                y_master = np.copy(y)
                flag = True
            else:
                x_master = np.concatenate((x_master, x))
                y_master = np.concatenate((y_master, y))
    x_master, y_master = remove_hours(x_master, y_master)
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_master, y_master, patient_id, output_folder)
    # Downsample the data
    X_train, y_train = downsample_shuffle_split(X_train, y_train)
    X_test, y_test = downsample_shuffle_split(X_test, y_test)

    return X_train, X_test, y_train, y_test


def train_test_split(X, y, patient_id, save_folder, split=0.8):
    """ Split the data into train and test sets for a given patient
    Parameters
    ----------
    X: np.array
        Data
    y: np.array
        Labels
    patient_id: int
        Number of the patient
    save_folder: str
        Path to the folder where to save the data
    split: float
        Percentage of the data to use for training
    Returns
    -------
    X_train: np.array
        Training data
    X_test: np.array
        Test data
    y_train: np.array
        Training labels
    y_test: np.array
    """
    patient_file = f"chb0{patient_id}" if patient_id < 10 else f"chb{patient_id}"
    if not os.path.exists(save_folder + patient_file):
        os.makedirs(save_folder + patient_file)
    # Split the data into train and test sets
    X_train = X[:int(split * X.shape[0])]
    X_test = X[int(split * X.shape[0]):]
    y_train = y[:int(split * y.shape[0])]
    y_test = y[int(split * y.shape[0]):]
    return X_train, X_test, y_train, y_test


def downsample_shuffle_split(X, y):
    """ Downsample X, y by the number of patients and shuffle the data
    Parameters
    ----------
    X: np.array
        Data
    y: np.array
        Labels
    Returns
    ----------
    X_downsampled: np.array
        Downsampled data
    y_downsampled: np.array
        Downsampled labels
    """
    np.random.seed(2102)
    random_indices = np.random.choice(
        X.shape[0], X.shape[0] // PATIENTS_SIZE, replace=False)
    return X[random_indices], y[random_indices]


def download_file(source, destination, force_download=False):
    """ Download a file from a source to a destination
    Parameters
    ----------
    source: str
        Source of the file
    destination: str
        Destination of the file
    force_download: bool
        if True, force the download of the file
    """
    if os.path.exists(destination) and not force_download:
        print("File " + destination + " already exists.")
        return
    print("Downloading " + source + " to " + destination + "...")
    urllib.request.urlretrieve(source, destination)
    print("Downloaded " + source + ".")


def normalize_data_and_save(X_train, y_train, X_test, y_test, dataset_folder):
    """ Normalize the data and save it
    Parameters
    ----------
    X_train: list of np.array
        Training data list
    y_train: list of np.array
        Training labels list
    X_test: list of np.array
        Test data list
    y_test: list of np.array
        Test labels list
    dataset_folder: str
        Path to the folder where to save the data
    """
    # Select 3 patients at random to remove from the training set
    patients_to_remove_train = np.random.choice(
        len(X_train), PATIENTS_SIZE // 4, replace=False)
    # Splice patients_to_remove_train indexes from the training set X and y
    X_train = [X_train[i]
               for i in range(len(X_train)) if i not in patients_to_remove_train]
    y_train = [y_train[i]
               for i in range(len(y_train)) if i not in patients_to_remove_train]
    # Find minimum length of the data for training set
    train_min_len = min([data.shape[0] for data in X_train])
    # Trim training set data to minimum length
    X_train = [data[:train_min_len] for data in X_train]
    y_train = [data[:train_min_len] for data in y_train]

    # Find minimum length of the data for test set
    test_min_len = min([data.shape[0] for data in X_test])
    # Trim test set data to minimum length
    X_test = [data[:test_min_len] for data in X_test]
    y_test = [data[:test_min_len] for data in y_test]

    # Concatenate training and test sets
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # Save preprocessed data to disk
    np.save(dataset_folder + "X_train.npy", X_train)
    np.save(dataset_folder + "y_train.npy", y_train)
    np.save(dataset_folder + "X_test.npy", X_test)
    np.save(dataset_folder + "y_test.npy", y_test)


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
    # file paths
    seizure_summary = eeg_database_folder + "seizure_summary.csv"
    records_summary = eeg_database_folder + "RECORDS"
    dataset_folder = 'dataset/'
    website = "https://archive.physionet.org/pn6/chbmit/"
    summary = "https://raw.githubusercontent.com/NeuroSyd/Integer-Net/master/copy-to-CHBMIT/seizure_summary.csv"
    # create required folders
    os.makedirs(eeg_database_folder, exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)
    # download seizure summary and records files if they don't exist
    if not os.path.exists(records_summary):
        urllib.request.urlretrieve(website + "RECORDS", records_summary)
    if not os.path.exists(seizure_summary):
        urllib.request.urlretrieve(summary, seizure_summary)
    # For each patient we are interested in
    patients = {f"chb{i:02d}": i for i in PATIENTS_LIST}
    # Read the records summary file to get the list of files
    with open(records_summary) as f:
        summary = f.read().splitlines()
    X_train, y_train, X_test, y_test = [], [], [], []
    for patient in patients.keys():
        # Get the list of files for the current patient
        patient_files = [file for file in summary if patient in file]
        # Create patient folder
        os.makedirs(eeg_database_folder + patient, exist_ok=True)
        # Download the files using multiprocessing -> if an error occurs, retry
        while True:
            try:
                print("Downloading " + patient + "...")
                with Pool(cpu_count()) as p:
                    p.starmap(download_file, [(website + file, eeg_database_folder + file, force_download)
                                              for file in patient_files])
            except Exception as e:
                print(type(e))
                print(
                    f"An error occurred while downloading a file of {patient}. Retrying...")
            else:
                break
        # Preprocess the data
        print("Preprocessing " + patient + "...")
        X_train_patient, X_test_patient, y_train_patient, y_test_patient = preprocess_to_numpy(records_summary,
                                                                                               seizure_summary,
                                                                                               eeg_database_folder,
                                                                                               patients[patient],
                                                                                               dataset_folder,
                                                                                               1)
        X_train.append(X_train_patient)
        X_test.append(X_test_patient)
        y_train.append(y_train_patient)
        y_test.append(y_test_patient)
        if remove and os.path.exists(eeg_database_folder + patient):
            shutil.rmtree(eeg_database_folder + patient)
    # Normalize the data and save it
    print("Normalizing data and saving it...")
    normalize_data_and_save(X_train, y_train, X_test,
                            y_test, dataset_folder)
    print("Data saved to disk.")
    if remove and os.path.exists(eeg_database_folder):
        shutil.rmtree(eeg_database_folder)


if __name__ == "__main__":
    folder = "chb-mit-scalp-eeg-database-1.0.0/"
    download_dataset(folder)
