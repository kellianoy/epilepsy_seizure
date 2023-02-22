import rampwf as rf
import numpy as np
import pandas as pd
from rampwf.utils.importing import import_module_from_source
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
import os

problem_title = 'Predection of seizure'

_prediction_label_name = [0, 1] # TODO

Predictions = rf.prediction_types.make_multiclass(label_names=_prediction_label_name*1)

class SeizureClassifierWorkflow:
    def __init__(self, workflow_element_names=['seizure_classifier.py']):
        self.workflow_element_names = workflow_element_names
        self.estimator = None

    def train_submission(self, module_path, X_train, y_train, train_is=None):
        estimator_module = import_module_from_source(
            os.path.join(module_path, self.workflow_element_names[0]),
            self.workflow_element_names[0],
            sanitize=True)

        model = estimator_module.SeizureClassifier()
        model.fit(X_train, y_train)
        
        return model
    
    def test_submission(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred
    
workflow = SeizureClassifierWorkflow()
score_types = [
    rf.score_types.Accuracy(name='acc'),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)

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
    path = "./dataset"
    # Check that the path exists
    if not os.path.exists(path):
        raise ValueError('The path {} does not exist'.format(path))
    # if not len(patients):
    #     raise ValueError('No patient specified')
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
