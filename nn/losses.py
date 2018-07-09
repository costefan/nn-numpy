import numpy as np


def cross_entropy(y, y_pred):

    return - np.sum(y * np.log(y_pred))
