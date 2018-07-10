import numpy as np
from config.logger import logger


def tanh(X):
    logger.info('Dim before tanh: {}'.format(X.shape))
    res_tanh = np.tanh(X)
    logger.info('Dim after tanh: {}'.format(res_tanh.shape))

    return res_tanh


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def relu(X):
    logger.info('Dim before relu: {}'.format(X.shape))
    res_relu = np.maximum(X, 0, X)
    logger.info('Dim after relu: {}'.format(res_relu.shape))

    return res_relu


def relu_deriv(x):
    return np.array(x > 0, dtype=np.int8)


def softmax(X):
    total_exp = np.sum(np.exp(X), axis=0)

    return np.exp(X) / total_exp
