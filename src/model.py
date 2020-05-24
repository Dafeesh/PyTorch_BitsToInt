'''
Description:
    Run this file to create a new PyTorch Neural Network model to 
     classify binary input as an integer.

Author:
    Blake Scherschel
'''
import os
from typing import Callable
import torch
import torch.nn.functional as F

from data import get_bits_to_number_training_data

# How many binary digits should the model support?
NUM_BIT_DIGITS = 4
assert NUM_BIT_DIGITS <= 8, '''
    (2^digits) would be the number of output nodes produced. Maybe pick a smaller number? " \
    Or remove this check if you are feeling spicy.
'''

# How many times should the training data be sent through the model during training?
NUM_EPOCHS = 100

# How sensitive should the model's weights be to change?
LEARNING_RATE = 0.001

class Model(torch.nn.Module):
    FILE_PATH = "./model.pytorch"

    def __init__(self):
        super().__init__()
        num_class_nodes = 2**NUM_BIT_DIGITS
        self.fc1 = torch.nn.Linear(NUM_BIT_DIGITS, num_class_nodes)
        self.fc2 = torch.nn.Linear(num_class_nodes, num_class_nodes)
        self.fc3 = torch.nn.Linear(num_class_nodes, num_class_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def create_model() -> Model:
    '''
    Create, train, and save a new model.
    '''
    model = Model()
    trainingsets = get_bits_to_number_training_data(digits=NUM_BIT_DIGITS)
    learning_rate = LEARNING_RATE

    train_model(
        model,
        trainingsets,
        criterion = torch.nn.functional.mse_loss,
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate),
        num_epochs = NUM_EPOCHS,
    )

    torch.save(model.state_dict(), Model.FILE_PATH)
    return model

def train_model(
    model: Model,
    trainingsets,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int
):
    '''
    Trains `model` against `trainingsets` with specified parameters.
    '''
    print("Training...")
    with open('loss_history.csv', 'wt') as loss_history_file:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch} of {num_epochs}...")
            for _number_in, _in, _out in trainingsets:
                model.zero_grad()
                attempt_output = model(_in)
                loss = criterion(attempt_output, _out)
                loss.backward()
                optimizer.step()

                loss_factor = float(loss)
                loss_history_file.write(f"{loss_factor}\n")

def get_model() -> Model:
    '''
    Returns a trained Model loaded from disk.
    If the file does not already exist, train and save the model first.
    '''
    if os.path.exists(Model.FILE_PATH):
        print("Loading model from file....")
        model = Model()
        model.load_state_dict(torch.load(Model.FILE_PATH))
        model.eval()
        print("...model loaded from file.")
        return model

    print("Model file does not exist. Training...")
    model = create_model()
    print("...successfully created and trained a new model!")
    return model

if __name__ == "__main__":
    create_model()
