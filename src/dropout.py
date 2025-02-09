import numpy as np

class dropout:
    '''
    Class that contains the implementation of a dropout layer

    Attributes:
        dropout_rate (float): Probability that a node will be deactivated
        mask (np.ndarray): Which neurons will be activated and deactivated
    '''

    def __init__(self, dropout_rate: float = 0.5) -> None:
    # Initializes the dropout layer with the given parameters
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the dropout function    

        # Only deactivates neurons during training
        if training:
            
            # Creates a mask to multiply by the input which deactivates a probability of neurons
            self.mask = (np.random.rand(*input_tensor.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return input_tensor * self.mask
        
        return input_tensor
    
    def backward(self, output_gradient: np.ndarray, _: float) -> np.ndarray:
    # Calculates the gradient of the previous layer

        # Only activated neurons affect on the output
        return output_gradient * self.mask
