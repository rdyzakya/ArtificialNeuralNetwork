from src.ANN import *
import numpy as np

if __name__ == '__main__':
    # Model Sigmoid
    model = Sequential()
    model.add(Dense(units=2, input_dim=2, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))

    