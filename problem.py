import rampwf as rf
import pandas as pd
from rampwf.utils.importing import import_module_from_source
import os

problem_title = 'Predection of seizure'

_prediction_label_name = [1, 2] # TODO

Prediction = rf.prediction_types.make_multiclass(label_names=_prediction_label_name*[1])

class SeizureClassifierWorkflow:
    def __init__(self, workflow_element_names=['seizure_classifier.py']):
        self.workflow_element_names = workflow_element_names
        self.estimator = None

    def train_submission(self, module_path, X_train, y_train):
        estimator_module = import_module_from_source(
            os.path.join(module_path, self.workflow_element_names[0]),
            self.workflow_element_names[0],
            sanitize=True)


        # TODO
        pass


workflow = rf.workflows.Estimator() # TODO

score_types = [
    rf.score_types.roc_auc(name='auc'),
    rf.score_types.Accuracy(name='acc'),
] # TODO

def get_cv(X, y):
    # TODO
    pass

def _read_data(path, f_name):
    fname = f"data_{f_name}"
    data = pd.read_csv(os.path.join(path, 'data', fname), index_col=0)

    fname = f"labels_{f_name}"
    labels = pd.read_csv(os.path.join(path, 'data', fname), index_col=0)

    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_SMALL = 100 # TODO change
        data = data.iloc[:N_SMALL]
        labels = labels.iloc[:N_SMALL]
    
    return data, labels

def get_train_data(path='.'):
    return _read_data(path, 'train.csv')

def get_test_data(path='.'):
    return _read_data(path, 'test.csv')