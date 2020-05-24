'''
Description:
    Run this file to create a new PyTorch Neural Network model to
     classify binary input as an integer. Currently only 4-bits (for simplicity).

    "But Blake, a neural network is a terrible option to classify binary numbers!"
    "Yes, this was an excercise for me to learn how to create neural networks in PyTorch."

Author:
    Blake Scherschel
'''
import os
from typing import Callable
import torch
import torch.nn.functional as F

from data import get_bits_to_number_training_data

MODEL_FILE = "./model.pytorch"

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 16)

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
    trainingsets = get_bits_to_number_training_data()
    learning_rate = 0.001

    train_model(
        model,
        trainingsets,
        criterion = torch.nn.functional.mse_loss,
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate),
        num_epochs = 100,
    )

    torch.save(model.state_dict(), MODEL_FILE)
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
    if os.path.exists(MODEL_FILE):
        print("Loading model from file....")
        model = Model()
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        print("...model loaded from file.")
        return model

    print("Model file does not exist. Training...")
    model = create_model()
    print("...successfully created and trained a new model!")
    return model

if __name__ == "__main__":
    create_model()
