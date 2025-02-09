import numpy as np

def activation_func(activation):
# Maps the user entered activation function to the written class
    activation_map = {
        None: NoActivation(),
        'relu': reLU(),
        'sigmoid': sigmoid(),
        'softmax': softmax(),
    }

    return activation_map[activation]

class NoActivation:
    def __init__(self) -> None:
    # Initalises the activation layer if none is selected
        pass

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
    # Returns the input if no activation is selected
        return input_tensor
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    # Returns the input if no activation is selected
        return output_gradient

class reLU:
    def __init__(self) -> None:
    # Initalises the reLU activation layer 
        self.input_tensor = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
    # Applies the reLU activation function to the input tensor
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    # Calculates the derivative of the reLU function and multiplies it by the gradient
        input_gradient = np.multiply(np.where(self.input_tensor > 0, 1, 0), output_gradient)
        return input_gradient
    
class sigmoid:
    def __init__(self) -> None:
    # Initalises the sigmoid activation layer 
        self.input_tensor = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
    # Applies the sigmoid activation function to the input tensor
        self.input_tensor = 1 / (1 + np.exp(-input_tensor))
        return self.input_tensor
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    # Calculates the derivative of the sigmoid function and multiplies it by the gradient
        input_gradient = np.multiply(self.input_tensor * (1 - self.input_tensor), output_gradient)
        return input_gradient
    
class softmax:
    def __init__(self) -> None:
    # Initalises the softmax activation layer 
        self.input_tensor = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
    # Applies the softmax activation function to the input tensor
        input_exp = np.exp(input_tensor)
        self.input_tensor = input_exp / np.sum(input_exp, axis=-1, keepdims=True)
        return self.input_tensor
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    # Calculates the derivative of the softmax function and multiplies it by the gradient
        
        # Expand dimensions to create a 3D array 
        self.input_tensor = self.input_tensor[..., np.newaxis] 
        
        # Create the Jacobian matrix 
        identity_matrix = np.eye(self.input_tensor.shape[1]) 
        jacobian_matrix = self.input_tensor * (identity_matrix - self.input_tensor.transpose((0, 2, 1))) 
        
        # Compute the gradient input and reshapes to the correct output
        input_gradient = jacobian_matrix @ output_gradient[..., np.newaxis]
        input_gradient = input_gradient.squeeze(-1)  
        
        return input_gradient
