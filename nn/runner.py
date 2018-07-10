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

        size_X = 100

        model = AlaResNet(n_input, n_hidden_1, n_hidden_2,
                          n_output, learning_rate)

        for epoch in range(n_epochs):
            predicted = 0
            for index in range(0, size_X, batch_size):
                X_batch = X[index:min(index + batch_size, size_X), :]
                y_batch = y[index:min(index + batch_size, size_X), :]

                out1, out2, y_pred = model.forward(X_batch)
                loss = cross_entropy(y_batch, y_pred)
                logger.info('Loss on iter {}: {}'.format(index, loss))

                # update weights using vanilla gradient descent
                d_out = (y_batch.flatten() - y_pred).reshape(n_output, 1)
                d_l2 = np.dot(np.diag(relu_deriv(model.a_2)),
                              np.dot(model.W_3.T, d_out))

                d_l1 = np.dot(np.diag(tanh_deriv(model.a_1)),
                              np.dot(model.W_2.T, d_l2))

                delta_W_out = np.dot(d_out,
                                     model.a_2.reshape((n_hidden_2, 1)).T)
                delta_W_two = np.dot(d_l2,
                                     model.a_1.reshape((n_hidden_1, 1)).T)
                delta_W_one = np.dot(d_l1, X_batch)

                logger.info(delta_W_out)
                logger.info(delta_W_two)
                logger.info(delta_W_one)

                # Update weights
                model.W_3 = model.W_3 - learning_rate * delta_W_out
                model.W_2 = model.W_2 - learning_rate * delta_W_two
                model.W_1 = model.W_1 - learning_rate * delta_W_one

                if y_batch.flatten().argmax(axis=0) == y_pred.argmax(axis=0):
                    predicted += 1

            accuracy = predicted / (X.shape[0] / batch_size)
            logger.info('Accuracy on epoch {}: {}'.format(epoch, accuracy))
