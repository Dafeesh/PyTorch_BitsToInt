'''
Description:
    Run this file to test an existing model saved to disk.
    The model should be able to take each number (0 -> N-1) and
     determine the correct integer.

    "But Blake, a neural network is a terrible option to classify binary numbers!"
    "Yes, this was an excercise for me to learn how to create neural networks in PyTorch."

Author:
    Blake Scherschel
'''
import torch

from model import get_model, NUM_BIT_DIGITS
from data import get_bits_to_number_training_data

def run_basic_test():
    '''
    Run through each number 0 -> N-1 and check that the model
      can interpret the bits of number as the correct number.
    '''
    trainingsets = get_bits_to_number_training_data(digits=NUM_BIT_DIGITS)
    model = get_model()

    correct = 0
    total = 0
    with torch.no_grad():
        for _number_in, _in, _out in trainingsets:
            output = model(_in)
            output_number = int(torch.argmax(output))
            print(f"Input={_number_in} Output={output_number}")
            if output_number == _number_in:
                correct += 1
            else:
                print(f"Oops! Model interpreted the bits of '{_number_in}' as '{output_number}'.")
            total += 1

    print('')
    print(f"Number of binary digits: {NUM_BIT_DIGITS}")
    print(f"{correct} out of {total} correct. {'%.1f'%(correct/total*100)}%")

if __name__ == "__main__":
    run_basic_test()
