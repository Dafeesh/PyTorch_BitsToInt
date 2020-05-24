# PyTorch - Bits To Integer

## What
*Neural network to classify binary digits as an integer*

This package includes a bare-bones example of how to classify a binary number using a neural network with the PyTorch library.
- You (probably): "But Blake, a neural network is a terrible option for classifying binary numbers!"
- Me: "Well, yea...this was an excercise for me to learn how neural networks are created in PyTorch. Maybe it can help you, too."

Check out these files:
- [src/learn.py](src/learn.py)
  - ```create_model()```
    - Where the model is created and trained with determined parameters.
    - Modify this file to play around with how the neural network is shaped and trained.
    - Running ```./run.sh``` will re-run the training and overwrite the existing model saved to file.
- [src/data.py](src/data.py)
  - ```NUM_BIT_DIGITS```
    - Specify a different number of digits.
    - WARNING: Output nodes are mapped 1:1 with all possible values so 2\*\*4==16 output nodes. 2\*\*16==65536 output nodes, so stick with smaller amounts like 4-8.

The neural network model is defined in [learn.create_model](src/learn.py). 

## How to run
First, install Python3.5+ and PyTorch (I'm using v1.5.0)
```
python3 --version
pip3 install pytorch
```
Then run the script!
```./run.sh```
Results are printed to stdtout, and two new files will be created:
- loss_history.csv
  - History of the loss factor for each iteration of training.
- model.pytorch
  - The trained model saved as a file.
  - You can now run ```./test.sh``` instead of ```./run.sh``` to avoid creating the model again.
