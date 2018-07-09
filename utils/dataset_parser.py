
import os

import numpy as np
import pandas as pd

DATASET_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'dataset')

DATASETS = {
    'spambase': 'spambase/spambase.csv',
    'dummy': 'dummy.csv'
}


def parse_dataset(dataset_name='spambase'):
    dataset_path = DATASETS[dataset_name]

    df = pd.read_csv(os.path.join(DATASET_DIRECTORY, dataset_path),
                     header=None, names=range(7))

    X = df.iloc[:, :-1].values

    y = pd.get_dummies(df.iloc[:, -1:]).values
    y = np.hstack((y, 1 - y))

    return X, y
