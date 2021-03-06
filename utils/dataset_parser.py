
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

DATASET_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'dataset')

DATASETS = {
    'spambase': {
        'name': 'spambase/spambase.csv',
        'range': 58},
    'dummy': {
        'name': 'dummy.csv',
        'range': 6},
    'hayes_roth': {
        'name': 'hayes-roth.csv',
        'range': 5
    }
}


def parse_dataset(dataset_name='spambase'):
    dataset = DATASETS[dataset_name]

    df = pd.read_csv(os.path.join(DATASET_DIRECTORY, dataset['name']),
                     header=None, names=range(dataset['range']))

    X = df.iloc[:, :-1].values

    y = pd.get_dummies(df.iloc[:, -1:]).values
    y = np.hstack((y, 1 - y))
    X = scale(X)

    return X, y
