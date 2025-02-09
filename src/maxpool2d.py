import numpy as np

class maxpool2d: 
    '''
    Class that contains the implementation of a maxpool layer

    Attributes:
        pool_size (tuple): Size of the pooling window
        strides (int): Strides of the max pooling
        padding (str): Padding type ('valid' or 'same')
        output_shape (tuple): Expected output shape to max pooling
        padding_amount (tuple): Amount of padding added to each x and y
        input_tensor (np.ndarray): Input to the max pooling layer
        max_positions (np.ndarray): Index of the largest item in each pool window 
    '''
    
    def __init__(self, pool_size: tuple = (3,3), strides: int = 1, padding: str = 'valid') -> None:
    # Initalises all base parameters for the max pooling layer object
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.strip().lower()

    def init_params(self, input_size: tuple) -> None:
    # Included for consistency to call the function to calculate expected output size
        self.output_shape, self.padding_amount = self.calculate_size(input_size)

    def calculate_size(self, input_size: tuple) -> tuple:
    # Calculates the expected output size and the amount of padding required on each x and y
        
        if self.padding == 'valid':
            output_width = int(np.ceil((input_size[1] - self.pool_size[0] + 1) // self.strides))
            output_height = int(np.ceil((input_size[2] - self.pool_size[1] + 1) // self.strides))
            return (input_size[0], output_height, output_width, input_size[3]), (0, 0)
    
        elif self.padding == 'same': 
            output_height = int(np.ceil(input_size[1] - self.pool_size[0] + 1 // self.strides))
            output_width = int(np.ceil(input_size[2] - self.pool_size[1] + 1 // self.strides))
            pad_width = int(((output_width - 1) * self.strides + self.pool_size[0] - input_size[1]) // 2)
            pad_height = int(((output_height - 1) * self.strides + self.pool_size[1] - input_size[2]) // 2)
 
            return (input_size[0], output_height, output_width, input_size[3]), (pad_width, pad_height)
    
    def forward(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
    # Performs the max pooling operation on the input tensor

        # Stores the input for backpropagation and pads it correctly
        self.input_tensor = input_tensor
        input_tensor = np.pad(input_tensor, pad_width=((0, 0), (self.padding_amount[0], self.padding_amount[0]), (self.padding_amount[1], self.padding_amount[1]), (0, 0)))     

        # Creates a sliding window view of the input tensor image and applies the strides
        image_patches = np.lib.stride_tricks.sliding_window_view(input_tensor, self.pool_size, axis=(1, 2))
        image_patches = image_patches[:, ::self.strides, ::self.strides, :, :, :]

        # Performs the max pooling over the items in each pool window
        image_patches = image_patches.reshape(self.input_tensor.shape[0], *self.output_shape[1:], -1)
        output_tensor = np.amax(image_patches, axis=-1, keepdims=False)
        
        if training:
            # Stores the index of the largest item in each pool window for back propagation
            self.max_positions = np.argmax(image_patches, axis=-1, keepdims=False)
               
        return output_tensor

    def backward(self, output_gradient: np.ndarray, _: float) -> np.ndarray:
    # Calculates the input gradient of the previous layer 

        # Creates a new tensor to hold the input gradient
        input_gradient = np.zeros_like(self.input_tensor)
        output_shape = output_gradient.shape

        # Reshapes the output gradient and max_positions tensor into compatible shapes
        output_gradient = output_gradient.reshape(output_shape[0], -1)
        max_positions = self.max_positions.reshape(output_shape[0], -1)

        # Converts the indexes from the max_positions format into indexes for input gradient
        input_gradient[np.unravel_index(max_positions, input_gradient.shape)] += output_gradient

        return input_gradient
