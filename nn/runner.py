from nn.ala_resnet import AlaResNet
from nn.losses import *
from .activations.functions import tanh_deriv, relu_deriv

from config.logger import logger


class Runner:

    @classmethod
    def run(cls, X, y):
        n_epochs = 1
        learning_rate = 0.01
        batch_size = 1

        n_classes = y.shape[1]

        n_input = X.shape[1]
        n_hidden_1 = 100
        n_hidden_2 = 200
        n_output = n_classes

        model = AlaResNet(n_input, n_hidden_1, n_hidden_2,
                          n_output, learning_rate)

        for epoch in range(n_epochs):
            predicted = 0
            for index in range(0, X.shape[0], batch_size):
                X_batch = X[index:min(index + batch_size, X.shape[0]), :]
                y_batch = y[index:min(index + batch_size, X.shape[0]), :]
                out1, out2, y_pred = model.forward(X_batch)
                loss = cross_entropy(y_batch, y_pred)
                logger.info('Loss on iter {}: {}'.format(index, loss))

                # update weights using gradient descent
                d_out = y_batch.flatten() - y_pred
                model.W_3 = model.W_3 - learning_rate * (
                    np.dot(out2.reshape(n_hidden_2, 1), d_out.reshape(1, 2))
                )

                d_l2 = np.diag(relu_deriv(model.a_2))\
                    .dot(model.W_3)\
                    .dot(d_out)
                model.W_2 = model.W_2 - learning_rate * d_l2 * out2

                d_l1 = np.diag(tanh_deriv(model.a_1)
                               .dot(model.W_2)
                               .dot(d_l2.reshape(n_hidden_2, 1)))
                model.W_1 = model.W_1 - learning_rate * d_l1 * X_batch.T
                # logging
                if y_batch.flatten().argmax(axis=0) == y_pred.argmax(axis=0):
                    predicted += 1

            accuracy = predicted / (X.shape[0] / batch_size)
            logger.info('Accuracy on epoch {}: {}'.format(epoch, accuracy))
