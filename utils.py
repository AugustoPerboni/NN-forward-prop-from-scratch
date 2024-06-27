import numpy as np 

def load_data():
    """
    Return first 1000 values from data saved from MNIST
    """
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
