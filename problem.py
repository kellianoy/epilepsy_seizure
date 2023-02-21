import rampwf as rf
import numpy as np
import os

problem_title = 'Predection of seizure'

_prediction_label_name = [1, 2, 3, 4]  # TODO

Prediction = rf.prediction_types.make_multiclass(
    label_names=_prediction_label_name)

workflow = rf.workflows.Estimator()  # TODO

score_types = []  # TODO


def get_cv(X, y):
    # TODO
    pass


def _read_data(path, dataset):
    """Read data from the numpy arrays
    Parameters
    ----------
    path: str
        The path to the data folder
    dataset: str
        The name of the dataset to load - either 'train' or 'test'
    Returns
    ----------
    X: np.ndarray
        Data of the specified patients
    y: np.ndarray
        Labels of the specified patients
    """
    # Check that the path exists
    if not os.path.exists(path):
        raise ValueError('The path {} does not exist'.format(path))
    if not len(patients):
        raise ValueError('No patient specified')
    if dataset not in ['train', 'test']:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    X_filename, y_filename = f"X_{dataset}.npy", f"y_{dataset}.npy"
    X_path, y_path = os.path.join(
        path, X_filename), os.path.join(path, y_filename)
    # Check that the files exist
    if not os.path.exists(X_path):
        raise ValueError('The path {} does not exist'.format(X_path))
    if not os.path.exists(y_path):
        raise ValueError('The path {} does not exist'.format(y_path))
    # Load the data
    return np.load(X_path), np.load(y_path)


def get_train_data(path='./dataset'):
    return _read_data(path, 'train')


def get_test_data(path='./dataset'):
    return _read_data(path, 'test')
