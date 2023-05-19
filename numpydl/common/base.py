import numpy as np


class DenseLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def relu(self, inputs):
        """
        ReLU Activation Function
        """
        raise NotImplementedError

    def softmax(self, inputs):
        """
        Softmax Activation Function
        """
        raise NotImplementedError

    def relu_derivative(self, dA, Z):
        """
        ReLU Derivative Function
        """
        raise NotImplementedError

    def forward(self, inputs, weights, bias, activation):
        """
        Single Layer Forward Propagation
        """
        raise NotImplementedError

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        """
        Single Layer Backward Propagation
        """
        raise NotImplementedError
