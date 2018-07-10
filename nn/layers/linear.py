import numpy as np


class Linear:

    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def __call__(self, X, W, *args, **kwargs):
        X = X.flatten()

        return np.dot(W, X)
