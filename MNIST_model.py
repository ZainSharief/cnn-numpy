import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from src.train import *
import time

def project(x_train, y_train, x_val, y_val):

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
        batch_size=64,
        epochs=1
    )

def torch_copy(x_train, y_train, x_val, y_val):

    device = 'cpu'

    x_train = torch.tensor(x_train, dtype=torch.float32).transpose(1, -1)
    y_train = torch.tensor(y_train, dtype=torch.float32) 
    x_val = torch.tensor(x_val, dtype=torch.float32).transpose(1, -1)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        nn.Dropout(0.5),
        
        nn.Flatten(),
        
        nn.Linear(53824, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        
        nn.Linear(128, 10),
        nn.Softmax(dim=1)
    )

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    epochs = 1
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f'\r{i+1}/{60000//64}', end='')
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                _, actual = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == actual).sum().item()

        print(f"validation loss: {total_loss/len(train_loader):.4f}, "f"validation accuracy: {100 * correct / total:.4f}%")

def main():

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True
    )
    val_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True
    )

    x_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
    x_val, y_val = val_dataset.data.numpy(), val_dataset.targets.numpy()

    # Normalize and reshape
    x_train, x_val = x_train / 255.0, x_val / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)

    y_train = np.eye(10, dtype='uint8')[y_train]
    y_val = np.eye(10, dtype='uint8')[y_val]

    p_start = time.time()
    project(x_train, y_train, x_val, y_val)
    p_total = time.time() - p_start
    print(f'project took {p_total} seconds to complete.')

    t_start = time.time()
    torch_copy(x_train, y_train, x_val, y_val)
    t_total = time.time() - t_start
    print(f'torch_copy took {t_total} seconds to complete.')

if __name__ == '__main__':
    main()