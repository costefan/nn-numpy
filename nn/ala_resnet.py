from nn.layers import *
from nn.activations import *
from numpy.random import normal as norm


class AlaResNet:

    def __init__(self, n_input, n_hidden_1, n_hidden_2,  n_output, l_r):
        """ Init all needed for ResNet
        :param n_1: num neurons on input layer
        :param n_2: num neurons on second layer
        :param n_3: num neurons on output layer
        :param l_r: Learning rate
        """
        self.l1 = Linear(n_input, n_hidden_1)
        self.l2 = Linear(n_hidden_1, n_hidden_2)
        self.l3 = Linear(n_hidden_2, n_output)

        self.a_1 = None
        self.a_2 = None
        self.a_3 = None

        self.learning_rate = l_r

        self._W_1, self._W_2, self._W_3 = self.init_weights(
            n_input, n_hidden_1, n_hidden_2,  n_output)

    def init_weights(self, n_input, n_hidden_1, n_hidden_2, n_output) -> tuple:
        """Xavier initializer of weights
        :return: 
        """
        # truncnorm can be used
        W_1 = norm(loc=0, scale=1/((n_input + n_hidden_1) / 2),
                   size=(n_input, n_hidden_1))
        W_2 = norm(loc=0, scale=1/((n_hidden_1 + n_hidden_2) / 2),
                   size=(n_hidden_1, n_hidden_2))
        W_3 = norm(loc=0, scale=1/((n_hidden_2 + n_output) / 2),
                   size=(n_hidden_2, n_output))

        return W_1, W_2, W_3

    @property
    def W_1(self):
        return self._W_1

    @property
    def W_2(self):
        return self._W_2

    @property
    def W_3(self):
        return self._W_3

    @W_1.setter
    def W_1(self, val):
        self._W_1 = val

    @W_2.setter
    def W_2(self, val):
        self._W_2 = val

    @W_3.setter
    def W_3(self, val):
        self._W_3 = val

    def forward(self, X_b):
        _a_1 = self.l1(X_b, self.W_1)
        out1 = tanh(_a_1)
        _a_2 = self.l2(out1, self.W_2)
        out2 = relu(_a_2)
        _a_3 = self.l3(out2, self.W_3)
        out3 = softmax(_a_3)

        self.a_1 = _a_1
        self.a_2 = _a_2
        self.a_2 = _a_2

        return out1, out2, out3

    def backprop(self, loss):
        pass
