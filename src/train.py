import numpy as np
import time

# Imports all written layers
from src.conv2d import conv2d
from src.maxpool2d import maxpool2d
from src.dense import dense
from src.flatten import flatten
from src.dropout import dropout
from src.batchNormalisation import batchNormalisation

# Imports the activation and loss functions
from src.activation import *
from src.loss import *

def initialise(model, input_shape: tuple) -> None:
# Initialises the layer weights & biases and calculates the output sizes given the input shape
    
    # Adds a batch size to the input
    input_shape = (1, *input_shape)

    # Iterates through each layer in the model
    for layer in model:

        # Calls the initialise parameters function on layers which have parameters
        if isinstance(layer, conv2d) or isinstance(layer, dense) or isinstance(layer, maxpool2d) or isinstance(layer, batchNormalisation):
            layer.init_params(input_shape)

        # Passes the temporary input through the model to calculate the output shape
        input_tensor = np.zeros(input_shape)
        input_shape = layer.forward(input_tensor, True).shape

def train(model, train, validation, loss_function, learning_rate=0.01, learning_rate_scheduler=1, epochs=10, batch_size=32, shuffle=True):
# Function to train the model and test it on the validation data

    # Splits the datasets into data and annotations
    x_train, y_train = train
    x_val, y_val = validation

    for e in range(epochs):

        # Resets the loss, timer and acccuracy variables to 0
        correct = 0
        total = 0
        loss = 0
        time_elapsed = 0

        # Shuffles the training data with the same permutation
        if shuffle:
            permutation = np.random.permutation(x_train.shape[0])  
            x_train = x_train[permutation]
            y_train = y_train[permutation]
    
        # Splits up the data into arrays with the batch size entered
        x_train = np.array_split(x_train, np.ceil(x_train.shape[0]/batch_size), axis=0)
        y_train = np.array_split(y_train, np.ceil(y_train.shape[0]/batch_size), axis=0)

        for batch_num, (x_batch, y_batch) in enumerate(zip(x_train, y_train)):

            # Starts the timer at the start of the pass
            start_time = time.time()

            # Forward pass 
            for layer in model:
                x_batch = layer.forward(x_batch, True)

            # Calculates the loss
            loss += loss_function.forward(x_batch, y_batch)

            # Calculates the maxmimum argument in the x_batch and y_batch
            predictions = np.argmax(x_batch, axis=1) 
            true_classes = np.argmax(y_batch, axis=1)

            # Sums up the number of correct predictions
            correct += np.sum(predictions == true_classes)
            total += x_batch.shape[0]

            # Backward pass
            grad = loss_function.dervivative(x_batch, y_batch)
            for layer in reversed(model):
                grad = layer.backward(grad, learning_rate)

            # Ends the timer and calculates the time taken 
            end_time = time.time()
            time_elapsed += (end_time - start_time)

            # Calculates the average time per batch in order to calculate seconds, minutes and hours remaining
            time_remaining = int((time_elapsed/(batch_num+1)) * (len(x_train)-(batch_num+1)))
            hours = time_remaining // 3600
            minutes = str((time_remaining % 3600) // 60).zfill(2)
            seconds = str(time_remaining % 60).zfill(2)
            if hours: time_formatted = f'{hours}:{minutes}:{seconds}' 
            else: time_formatted = f'{minutes}:{seconds}' 

            print(f"\033[Kepoch={e + 1}/{epochs}: batch={batch_num+1}/{len(x_train)}, loss={loss/(batch_num+1):.5f}, accuracy={correct/total:.4f}, time remaining={time_formatted}", end="\r")
        
        # Unsplits the tensors so they can be re-shuffled
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
            
        # Validates the existance of validation data
        if x_val is not None:

            # Resets the loss to 0 for validation
            loss = 0
            correct = 0
            total = 0

            # Splits up the data into arrays with the batch size entered
            x_val = np.array_split(x_val, np.ceil(x_val.shape[0]/batch_size), axis=0)
            y_val = np.array_split(y_val, np.ceil(y_val.shape[0]/batch_size), axis=0)

            for batch_num, (x_batch, y_batch) in enumerate(zip(x_val, y_val)):

                # Forward pass 
                for layer in model:
                    x_batch = layer.forward(x_batch)

                # Calculates the loss
                loss += loss_function.forward(x_batch, y_batch)

                # Calculates the maxmimum argument in the x_batch and y_batch
                predictions = np.argmax(x_batch, axis=1) 
                true_classes = np.argmax(y_batch, axis=1)

                # Sums up the number of correct predictions
                correct += np.sum(predictions == true_classes)
                total += x_batch.shape[0]
            
            print(f"\nepoch={e + 1}/{epochs}: validation loss={loss/(len(x_val)):.5f}, validation accuracy={correct/total}")

            # Unsplits the tensors so they can be re-shuffled
            x_val = np.concatenate(x_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)

        else:
            print('\n')

        # Decreases the learning rate for better convergence
        learning_rate *= learning_rate_scheduler
