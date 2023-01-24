import os
import wfdb

# This is a script to download the data from CHB-MIT scalp EEG database
# The dataset can be found here: https://archive.physionet.org/pn6/chbmit/
# Or here: https://physionet.org/content/chbmit/1.0.0/
# Recordings, grouped into 23 cases, were collected from 22 subjects

# To download the data, we use the wfdb package as recommended by the authors
# The wfdb package can be found here: https://pypi.org/project/wfdb/

# Start of the script
# If folder dataset does not exist, create it
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')
# Download the dataset in the folder dataset with the annotations
wfdb.dl_database(db_dir='chbmit',
                 dl_dir='./dataset',
                 records='all')
# This seems to generate an error because of non-existing files:
# 404 Error: Not Found for url:
# https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf.hea
