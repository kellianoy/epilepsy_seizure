import numpy as np

class EegClassifier(object):
    def __init__(self):
        pass

    def fit(self, data, y):
        classes = [1,2]
        self.n_classes = len(classes)
        pass

    def predict(self, data):
        proba = np.random.rand(len(data), self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]
        return proba