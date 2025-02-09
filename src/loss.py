import numpy as np
from src.activation import softmax

class mse:

    def __init__(self) -> None:
    # Creates an instance of mean squared error
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the mean-squared error loss
        return np.mean(np.power(y_true - y_pred, 2))
    
    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the dervivative of mean-squared error loss
        return (2 * (y_pred - y_true)) / np.size(y_pred)

class binary_crossentropy:

    def __init__(self) -> None:
    # Creates an instance of the binary cross-entropy 
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the binary cross-entropy loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))    
    
    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the dervivative of binary cross-entropy loss
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    
class categorical_crossentropy: 
    
    def __init__(self) -> None:
    # Creates an instance of the softmax class to be applied 
        self.softmax = softmax()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the categorical crossentropy loss
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        cross_entropy = -np.sum(y_true * np.log(y_pred), axis=-1)
        return np.mean(cross_entropy)

    def dervivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Calculates the dervivative of categorical crossentropy loss
        y_pred = self.softmax.forward(np.clip(y_pred, 1e-7, 1-1e-7))
        gradient = y_pred - y_true 
        return gradient
