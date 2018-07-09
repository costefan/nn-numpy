import numpy as np
from config.logger import logger

# TODO: DECORATOR


def tanh(X):
    logger.info('Dim before tanh: {}'.format(X.shape))
    res_tanh = np.tanh(X)
    logger.info('Dim after tanh: {}'.format(res_tanh.shape))

    return res_tanh


def tanh_deriv(X):
    return 1 - (X ** 2)


def relu(X):
    logger.info('Dim before relu: {}'.format(X.shape))
    res_relu = np.maximum(X, 0, X)
    logger.info('Dim after relu: {}'.format(res_relu.shape))

    return res_relu


def relu_deriv(X):
    for i in range(0, len(X)):
        if X[i] > 0:
            X[i] = 1
        else:
            X[i] = 0
    return X


def softmax(X):
    logger.info('Dim before softmax: {}'.format(X.shape))
    e_x = np.exp(X)

    res_softmax = e_x / e_x.sum(axis=0)
    logger.info('Dim after softmax: {}'.format(res_softmax.shape))

    return res_softmax
