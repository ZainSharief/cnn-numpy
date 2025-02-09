import numpy as np
from src.activation import *

class dense:
    '''
    Class that contains the implementation of a dense layer

    Attributes:
        units (int): Number of units in the dense operation
        activation (str): Activation applied after the operation
        weight_params (np.ndarray): Trainable weight parameters
        bias_params (np.ndarray): Trainable bias parameters
        input_tensor (np.ndarray): Input array to the dense operation
        output (np.ndarray): Output array of completed dense operation
    '''

    def __init__(self, units: int, activation: str = None) -> None:
    # Initalises all base parameters for the dense layer object
        self.units = units
        self.activation = activation_func(activation)

    def init_params(self, input_size: tuple) -> None:
    # Initialises the weights and biases for the network

        # Calculates the expected output size
        self.output = self.calculate_size(input_size)

        # Uses Xavier Initialization to initialise the weights
        input_neurons = np.prod(input_size[1:])
        self.weight_params = np.random.randn(self.units, input_neurons) * np.sqrt(2 / (input_neurons + self.units))

        # Initialises the bias parameters as 0
        self.bias_params = np.zeros(self.units) 

    def calculate_size(self, input_size: tuple) -> None:
    # Calculates the expected output size
        return np.zeros([input_size[0], self.units])

    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the dense operation

        # Matrix multiplies the input and weights and adds on the bias
        self.input_tensor = input_tensor
        self.output = np.dot(input_tensor, self.weight_params.T) + self.bias_params

        # Applies the activation function to the input
        self.output = self.activation.forward(self.output)

        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    # Backwards function which updates the weight and bias parameters, calculates the input gradient for the next layer
        
        # Calculates the gradient of the activation function
        output_gradient = self.activation.backward(output_gradient)

        # Calculates the weight and input gradient 
        weights_gradient = np.dot(output_gradient.T, self.input_tensor)
        input_gradient = np.dot(output_gradient, self.weight_params)

        # Updates the weight and bias parameters accordingly 
        self.weight_params -= learning_rate * (weights_gradient / self.input_tensor.shape[0])
        self.bias_params -= learning_rate * np.mean(output_gradient, axis=0)

        return input_gradient
