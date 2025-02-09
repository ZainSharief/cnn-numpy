import numpy as np

class batchNormalisation:
    '''
    Class that contains the implementation of a batch normalisation layer

    Attributes:
        epsilon (float): Constant to avoid division of 0
        momentum (float): Change in the running mean and variance over mini-batches
        input_shape (tuple): Shape of the input
        num_channels (int): Number of channels in the input
        gamma (np.mdarray): Trainable parameter to act as a weight
        beta (np.mdarray): Trainable parameter to act as a bias
        running_mean (np.ndarray): Currently running mean over all batches
        running_variance (np.ndarray): Currently running variance over all batches
        X_centered (np.ndarray): Centered input data
        std_inv (np.ndarray): Inverse of the standard deviation calculation
        X_normalized (np.ndarray): Normalised input data
    '''

    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9) -> None:
    # Initializes the batch normalisation layer with the given parameters
        self.epsilon = epsilon
        self.momentum = momentum

    def init_params(self, input_size: tuple) -> None:
    # Initialises the gamma, beta parameters and the running mean and variance

        # Finds the number of channels in the input
        self.num_channels = input_size[-1]

        # Initalises empty beta & gamma parameters for each channel 
        self.gamma = np.ones(self.num_channels)
        self.beta = np.zeros(self.num_channels)

        # Initialises the running mean and variance
        self.running_mean = np.zeros(self.num_channels)
        self.running_variance = np.zeros(self.num_channels)

    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the batch normalisation operation

        # Checks if training to update the running mean and variance and calculate values for backpropagation
        if training:

            # Finds the input shape of the tensor
            self.input_shape = input_tensor.shape

            # Flattens the input for compatibility with 2D and 4D tensors
            flattened_input = input_tensor.reshape(-1, self.num_channels)

            # Calculates the mean and variance of the current batch
            mean = np.mean(flattened_input, axis=0, keepdims=True)
            variance = np.var(flattened_input, axis=0, keepdims=True)

            # Standardises the input tensor and uses the parameters to rescale and shift the output
            self.norm_tensor = (input_tensor - mean) / np.sqrt(variance + self.epsilon)
            output_tensor = (self.gamma * self.norm_tensor) + self.beta

            # Updates the running mean and variance using the momentum value
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * variance

            # Performs calculations required in backpropagation
            self.X_centered = flattened_input - mean
            self.std_inv = 1.0 / np.sqrt(variance + self.epsilon)
            self.X_normalized = self.X_centered * self.std_inv           
        
        else:

            # Standardises the input tensor using the running mean and variance 
            norm_tensor = (input_tensor - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)

            # Uses the parameters to rescale and shift the output
            output_tensor = (self.gamma * norm_tensor) + self.beta

        return output_tensor         

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    # Backwards function which updates the beta and gamma parameters, calculates the input gradient for the next layer 

        # Flattens the input for compatibility with 2D and 4D tensors
        flattened_output = output_gradient.reshape(-1, self.num_channels)
        flattened_norm = self.norm_tensor.reshape(-1, self.num_channels)

        # Calculates the derivative of gamma and beta with respect to the loss
        dgamma = np.sum(flattened_output * flattened_norm, axis=0)
        dbeta = np.sum(flattened_output, axis=0)

        # Calculates the derivative of the variance and mean 
        dX_norm = flattened_output * self.gamma
        dvar = np.sum(dX_norm * self.X_centered, axis=0) * -0.5 * self.std_inv**3
        dmean = np.sum(dX_norm * -self.std_inv, axis=0) + dvar * np.sum(-2.0 * self.X_centered, axis=0) / self.input_shape[0]

        # Inverses the standardisation to calculate the input gradient
        input_gradient = (dX_norm * self.std_inv + dvar * 2.0 * self.X_centered / self.input_shape[0] + dmean / flattened_output.shape[0]).reshape(self.input_shape)

        # Updates the beta and gamma parameters 
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return input_gradient
