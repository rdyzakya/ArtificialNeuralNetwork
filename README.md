# ArtificialNeuralNetwork
Tugas Besar Machine Learning

Contoh memakai kodenya:

```python
from src.ANN import *
import numpy as np


model = Sequential()
model.add(Dense(units=2, input_dim=2, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

# only one data point / 1 instance
input_matrix_1 = np.array([[0,1]])

# five data points / 5 instances
input_matrix_2 = np.array([[0,1],[1,0],[1,1],[0,0],[1,1]])

res_1 = model.forward_feed(input_matrix_1)

res_2 = model.forward_feed(input_matrix_2)

print("Result with one data point:")
print(res_1)
print("Result with five data points:")
print(res_2)

# modify weights and biases
weights_1 = np.array([[1,1],[1,1]])
biases_1 = np.array([1,1])
model.layers[0].compile_weight_and_bias(weights=weights_1,biases=biases_1)

weights_2 = np.array([[1],[1]])
biases_2 = np.array([1])
model.layers[1].compile_weight_and_bias(weights=weights_2,biases=biases_2)
```
