import numpy as np

class flatten:
    '''
    Class that contains the implementation of a flatten layer

    Attributes:
        input_shape (np.ndarray): Shape inputted to the flatten operation
    '''

    def __init__(self) -> None:
    # Creates an instance of the flatten layer
        self.input_shape = None

    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the flatten operation
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], -1))

    def backward(self, output_gradient: np.ndarray, _: float) -> np.ndarray:
    # Backwards function which returns the input to its original shape
        return np.reshape(output_gradient, self.input_shape)
