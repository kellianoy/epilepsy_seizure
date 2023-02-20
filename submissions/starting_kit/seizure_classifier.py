import numpy as np

class SeizureClassifier(object):
    def __init__(self):
        pass

    def fit(self, data: list, y):
        classes = [1,2]
        self.n_classes = len(classes)
        pass

    def predict(self, data: list):
        proba = np.random.rand(len(data), self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]
        return proba