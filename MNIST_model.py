import tensorflow as tf
from src.train import *

# Creates an example sequential model
model = [
    conv2d(filters=32, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
    batchNormalisation(),
    conv2d(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
    batchNormalisation(),
    maxpool2d(pool_size=(2,2), strides=1, padding='same'),
    dropout(dropout_rate=0.5),
    flatten(),
    dense(units=128, activation='relu'),
    batchNormalisation(),
    dropout(dropout_rate=0.5),
    dense(units=10, activation='softmax'),
]

# Extracts the MNIST dataset from tensorflow
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Preprocesses the dataset images
x_train, x_val = x_train / 255.0, x_val / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Preprocesses the dataset annotations
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Compiles the model, creating all variables
initialise(
    model=model, 
    input_shape=(28, 28, 1)
)

# Training the model
train(
    model=model,
    train=(x_train, y_train),
    validation=(x_val, y_val),
    loss_function=categorical_crossentropy(),
    learning_rate=0.01,
    learning_rate_scheduler=0.85,
    batch_size=32,
    epochs=30
)
