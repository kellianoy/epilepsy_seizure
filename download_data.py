import os
import numpy as np
from mne.io import read_raw_edf
import csv
import urllib.request

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
        data :  array of data from CHB-MIT dataset

        Returns
        ----------
        numpy array
    """

    fft_tensor = np.fft.rfft(data[:, 1:], axis=0)
    fft_tensor = np.float16(np.log10(np.abs(fft_tensor)+1e-6))
    indices = np.where(fft_tensor <= 0)
    fft_tensor[indices] = 0

    return fft_tensor  # ,freq_array,time_array


def edf_to_array(filename_in, seizures_time_code, time_lenght, number_of_patient):
    """


        Parameters
        ----------
        filename_in : str

        filename_out : str

        seizures_time_code : list

        time_lenght : int

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

    print(seizures_time_code)
    while time_iterator*1000 + time_lenght*1000 < tmp[-1, 0]:
        index_start = int(time_iterator*freq_mean)
        index_stop = int(index_start + time_lenght*freq_mean)
        data = tmp[index_start:index_stop, 1:]

        flag_ictal = 0

        for bounds in seizures_time_code:
            if (bounds[0] < tmp[index_start, 0]/1000 < bounds[1]) and (bounds[0] < tmp[index_stop, 0]/1000 < bounds[1]):
                flag_ictal = 1

        x.append(data)
        y.append([flag_ictal, tmp[index_start, 0]])
        time_iterator += time_lenght*(1-flag_ictal) + flag_ictal*2/(256)

    return np.array(x), np.array(y)


def preprocess_to_numpy(records_path, seizure_summary_path, database_path, number_of_patient, dir_where_to_save, time_lenght):
    """ Preprocess the data from the CHB-MIT dataset and save it in numpy format
    Parameters
    ----------
    records_path : str
        Path to the file containing the `RECORDS` file
    seizure_summary_path : str
        Name of the file containing the list of the filename of the data
    database_path : str
        Path to the folder containing the data
    """
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

    flag = 0
    for filename in csv_reader_list_filename:
        if int(filename[0][3]+filename[0][4]) == number_of_patient:
            bounds = []
            if filename[0][6:] in liste_bounds[0]:
                indices = [i for i, x in enumerate(
                    liste_bounds[0]) if x == filename[0][6:]]
                for indice in indices:
                    bounds.append([liste_bounds[1][indice],
                                  liste_bounds[2][indice]])
            print(filename)
            x, y = edf_to_array(database_path +
                                filename[0], bounds, time_lenght, number_of_patient)
            if flag == 0:
                x_master = np.copy(x)
                y_master = np.copy(y)
                flag = 1
            else:
                x_master = np.concatenate((x_master, x))
                y_master = np.concatenate((y_master, y))

    patient_file = "chb0" + \
        str(number_of_patient) if number_of_patient < 10 else "chb" + \
        str(number_of_patient)
    if not os.path.exists(dir_where_to_save + patient_file):
        os.makedirs(dir_where_to_save + patient_file)
    np.save(dir_where_to_save + patient_file + "/" + patient_file +
            "_X.npy", np.float16(x_master))
    np.save(dir_where_to_save + patient_file + "/" +
            patient_file + "_y.npy", np.float16(y_master))


def download_dataset(folder):
    """ Download the dataset from the website, limiting the number of records
    (some records are not working)
    Parameters
    ----------
    folder : str
        folder where to save the dataset
    """
    seizure_summary = folder + "seizure_summary.csv"
    records_summary = folder + "RECORDS"

    if not os.path.exists(folder):
        os.makedirs(folder)
    # website where to download the dataset
    website = "https://archive.physionet.org/pn6/chbmit/"
    summary = "https://raw.githubusercontent.com/NeuroSyd/Integer-Net/master/copy-to-CHBMIT/seizure_summary.csv"
    # download the records file (header)
    if not os.path.exists(records_summary):
        urllib.request.urlretrieve(website+"RECORDS", records_summary)
    # download the seizure summary file
    if not os.path.exists(seizure_summary):
        urllib.request.urlretrieve(summary, seizure_summary)
    # For each patient we are interested in, download the records
    for i in [2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23, 24]:
        # Retrieve summary of seizures
        current_patient = "chb0"+str(i) if i < 10 else "chb"+str(i)
        # read records pertaining to current patient and download the files
        if not os.path.exists(folder+current_patient):
            os.makedirs(folder+current_patient)
        with open(records_summary, "r") as f:
            records = f.readlines()
            for record in records:
                if record.startswith(current_patient) and not os.path.exists(folder+record.strip()):
                    urllib.request.urlretrieve(
                        website+record.strip(), folder+record.strip())
                    print("Downloaded "+record.strip())
        # Preprocess the data and save it in numpy format
        preprocess_to_numpy(records_summary,
                            seizure_summary,
                            folder,
                            i,
                            'dataset/',
                            1)


if __name__ == "__main__":

    folder = "chb-mit-scalp-eeg-database-1.0.0/"
    download_dataset(folder)
