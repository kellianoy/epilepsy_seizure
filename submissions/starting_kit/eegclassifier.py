import numpy as np

class EegClassifier(object):
    def __init__(self):
        pass

    def fit(self, data, y):
        classes = [0,1]
        self.n_classes = len(classes)

    def predict(self, data):
        proba = np.random.rand(len(data), self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]
        return proba