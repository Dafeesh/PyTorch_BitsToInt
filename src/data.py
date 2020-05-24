'''
Description:
    Helps generate datasets for classifying binary as an integer.
    Currently only 4-bits (for simplicity).

    "But Blake, a neural network is a terrible option to classify binary numbers!"
    "Yes, this was an excercise for me to learn how to create neural networks in PyTorch."

Author:
    Blake Scherschel
'''
import torch
from torch import FloatTensor
from typing import List, Tuple

NUM_BIT_DIGITS = 4

def get_list_of_bits(n, digits=NUM_BIT_DIGITS) -> List[int]:
    bits = list()
    for d in range(digits):
        bits.append(int(n >> d & 0x1))
    return bits

def get_nth_bit_list(n, size) -> List[int]:
    x = list()
    for i in range(size):
        x.append(int(i == n))
    return x

def get_bits_to_number_training_data() -> List[Tuple[int,FloatTensor,FloatTensor]]:
    datasets = list()
    for n in range(2**NUM_BIT_DIGITS):
        datasets.append((
            # Represented number
            n,
            # Represented number as a tensor of bits
            torch.FloatTensor(get_list_of_bits(n)), 
            # Represented number as an Nth activated Tensor. (ex. 3 -> [0, 0, 0, 1, 0, ...]
            torch.FloatTensor(get_nth_bit_list(n, 2**NUM_BIT_DIGITS))
        ))
    return datasets
