import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

        # Initialize random weights and biases
        self.w1 = np.random.uniform(-1.0, 1.0, size=(self.input_size, self.hidden_size))
        self.w2 = np.random.uniform(-1.0, 1.0, size=(self.hidden_size, self.output_size))
        self.b1 = np.random.uniform(-1.0, 1.0, size=(1, self.hidden_size))
        self.b2 = np.random.uniform(-1.0, 1.0, size=(1, self.output_size))

    def forward(self, inputs):
        predictions = []
        for input in inputs:
            x = input.reshape(1, input.shape[0])
            z1 = np.dot(x, self.w1) + self.b1
            h1 = self._tanh(z1)
            z2 = np.dot(h1, self.w2) + self.b2
            h2 = self._sigmoid(z2)
            y_hat = (h2 > 0.5).astype(int)
            predictions.append(y_hat[0][0])
        return predictions

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def _dropout(self, x, dropout_rate):
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
        output = x * mask
        return output

    def __deepcopy__(self, memo):
        copy_nn = type(self)(self.input_size, self.hidden_size)
        copy_nn.w1 = self.w1.copy()
        copy_nn.w2 = self.w2.copy()
        copy_nn.b1 = self.b1.copy()
        copy_nn.b2 = self.b2.copy()
        return copy_nn
