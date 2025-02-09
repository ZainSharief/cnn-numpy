import numpy as np
from src.activation import *

class conv2d:
    '''
    Class that contains the implementation of a convolutional layer

    Attributes:
        filters (int): Number of filters
        strides (int): Stride of the convolution
        padding (str): Padding type ('valid' or 'same')
        activation (str): Activation function name
        kernel_size (tuple): Size of the kernel (height, width)
        padding_amount (tuple): Amount of padding added to each x and y
        input_size (tuple): Shape of the input
        kernel_shape (tuple): Shape of the kernel
        kernel_params (np.ndarray): Trainable parameters in the kernel
        bias_params (np.ndarray): Trainable bias parameters 
        input_tensor (np.ndarray): Input to the convolution
        output (np.ndarray): Output of the convolution
    '''

    def __init__(self, filters: int, kernel_size: tuple = (3,3), strides: int = 1, padding: str = 'valid', activation: str = None) -> None: 
    # Initializes the convolutional layer with the given parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.strip().lower()
        self.activation = activation_func(activation)

    def init_params(self, input_size: tuple) -> None:
    # Initialises the kernel & bias parameters for the convolution given the expected output size

        # Calculates the expected output size
        self.input_size = input_size
        self.output_shape, self.padding_amount = self.calculate_size(input_size)

        # Uses Uniform Initialisation to generate the kernel parameters
        self.kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.input_size[3], self.filters)
        self.kernel_params = np.random.uniform(-0.5, 0.5, self.kernel_shape)

        # Initialises the bias parameters as 0
        self.bias_params = np.zeros(self.output_shape[1:])

    def calculate_size(self, input_size: tuple) -> tuple:
    # Calculates the expected output size and the amount of padding required on each x and y

        if self.padding == 'valid':
            output_width = np.ceil((input_size[1] - self.kernel_size[0] + 1) / self.strides).astype(int)
            output_height = np.ceil((input_size[2] - self.kernel_size[1] + 1) / self.strides).astype(int)
            return (input_size[0], output_width, output_height, self.filters), (0, 0)
        
        elif self.padding == 'same': 
            output_width = np.ceil(input_size[1] / self.strides).astype(int)
            output_height = np.ceil(input_size[2] / self.strides).astype(int)
            pad_width = np.ceil(((output_width - 1) * self.strides + self.kernel_size[0] - input_size[1]) / 2).astype(int)
            pad_height = np.ceil(((output_height - 1) * self.strides + self.kernel_size[1] - input_size[2]) / 2).astype(int) 

            return (input_size[0], output_height, output_width, self.filters), (pad_height, pad_width)

    def correlate(self, input_tensor: np.ndarray, kernel: np.ndarray, output_shape: tuple, strides: int = 1) -> np.ndarray:
    # Performs a correlation between a tensor and kernel

        # Creates a sliding window view of the input with the shape of the kernel
        image_patches = np.lib.stride_tricks.sliding_window_view(input_tensor, kernel.shape[:3], axis=(1, 2, 3))
        
        # Applies the strides to the image view 
        image_patches = image_patches[:, ::strides, ::strides, :, :, :, :]
        
        # Reshapes the image and kernel into the correct format for convolution
        image_patches = image_patches.reshape(input_tensor.shape[0], *output_shape[1:3], -1)
        kernel = kernel.reshape(-1, kernel.shape[3])
        
        # Convolutes the input tensor and kernel
        output_tensor = np.tensordot(image_patches, kernel, axes=([3], [0]))
        
        return output_tensor
    
    def fftconvolve(self, tensor1: np.ndarray, tensor2: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    # Performs a fast convolution between two tensors

        # Transforms the tensors into the frequency domain
        tensor1 = np.fft.fftn(tensor1, kernel_shape)
        tensor2 = np.fft.fftn(tensor2, kernel_shape)

        # Computes the inverse discrete fourier transform of the tensor
        output = tensor1 * tensor2
        output = np.fft.ifftn(output)

        # Extracts the real part of the inverse
        output = np.real(output)

        return output

    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the forward pass on the input tensor
        
        # Stores the input to be used in backpropagation
        self.input_tensor = input_tensor

        # Applies padding to the input tensor and correlates it with the kernel 
        input_tensor = np.pad(input_tensor, pad_width=((0, 0), (self.padding_amount[0], self.padding_amount[0]), (self.padding_amount[1], self.padding_amount[1]), (0, 0))) 
        output_tensor = self.correlate(input_tensor, self.kernel_params, self.output_shape, strides=self.strides)

        # Applies the activation function on the output tensor
        output_tensor = self.activation.forward(output_tensor)
        
        return output_tensor

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    # Backwards function which updates the weight and bias parameters, calculates the input gradient for the next layer

        # Calculates the gradient of the activation function
        output_gradient = self.activation.backward(output_gradient)
        output_shape = output_gradient.shape
        input_gradient = np.zeros(self.input_size)

        # Calculates the padding on the output gradient to match the input gradient
        pad_x = np.ceil((self.kernel_size[0] + input_gradient.shape[1] - output_gradient.shape[1] - 1) // 2).astype(int)
        pad_y = np.ceil((self.kernel_size[1] + input_gradient.shape[2] - output_gradient.shape[2] - 1) // 2).astype(int)

        # Pads the output gradient and transposes the kernel for a full convolution
        input_gradient = self.correlate(
            np.pad(output_gradient, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (0, 0))), 
            self.kernel_params.transpose((1, 0, 3, 2)), 
            input_gradient.shape
        )

        # Computes the gradient of the unstrided input tensor
        if self.strides > 1:
            output_strided = np.zeros((output_shape[0], output_shape[1] * self.strides, output_shape[2] * self.strides, output_shape[3]), dtype=np.float32)
            output_strided[:, 1::self.strides, 1::self.strides, :] = output_gradient
        else:
            output_strided = output_gradient

        # Calculates the kernel gradient using fast convolutions
        kernel_gradient = self.fftconvolve(self.input_tensor, output_strided, self.kernel_shape)

        # Updates the kernel and bias parameters           
        self.kernel_params -= learning_rate * (kernel_gradient / input_gradient.shape[0]) 
        self.bias_params -= learning_rate * np.mean(output_gradient, axis=0) 

        return input_gradient
