from operator import mod
from src.ANN import *
from src.dump import *
import numpy as np
import json

if __name__ == "__main__":
    input_single = np.array([[1, 1]])

    input_batch = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    output_batch = np.array([[1], [1], [0], [0]])

    # untuk mencoba input dari bacaan file (load file)
    # data = load("tes.json")
    # model1 = Sequential()
    # model1.useJSON(data)

    # Model Sigmoid
    # model = Sequential()
    # model.add(Dense(units=3, input_dim=2, activation_function="sigmoid"))
    # model.add(Dense(units=1, activation_function="sigmoid"))

    # weight_hidden_layer = np.array([[0.2, 0.3, -0.1], [0.3, 0.1, -0.1]])
    # bias_hidden_layer = np.array([-0.3, 0.3, 0.3])
    # model.layers[0].compile_weight_and_bias(
    #     weights=weight_hidden_layer, biases=bias_hidden_layer
    # )

    # weight_output_layer = np.array([[0.5], [-0.3], [-0.4]])
    # bias_output_layer = np.array([-0.1])
    # model.layers[1].compile_weight_and_bias(
    #     weights=weight_output_layer, biases=bias_output_layer
    # )

    # res_single = model.predict(input_single)

    # print(res_single)

    # model.compile("sum_squared_error", 0.2, 0.05)

    # y_true = np.array([[0]])
    # # model.error_term(y_true=y_true, y_pred=res_single)
    # model._backprop(input_single, y_true)
    # model.summary()

    # res_batch = model.predict(input_batch)

    # print(res_batch)

    # Model RelU + Linear
    model = Sequential()
    model.add(Dense(units=2, input_dim=2, activation_function="relu"))
    model.add(Dense(units=1, activation_function="linear"))
    model.compile("sum_squared_error", 0.02, 0.05)
    model.fit(input_batch, output_batch, 1)
    model.summary()
    print(model.predict(input_single))

    # weight_hidden_layer = np.array([[1, 1], [1, 1]])
    # bias_hidden_layer = np.array([0, -1])
    # model.layers[0].compile_weight_and_bias(weights=weight_hidden_layer, biases=bias_hidden_layer)

    # weight_output_layer = np.array([[1], [-2]])
    # bias_output_layer = np.array([0])
    # model.layers[1].compile_weight_and_bias(weights=weight_output_layer, biases=bias_output_layer)

    # res_single = model.predict(input_single)

    # print(res_single)

    # res_batch = model.predict(input_batch)

    # print(res_batch)

    # # dump(model,"tes.json")

    # # untuk mencoba dump dari load file
    # # dump(model1,"tes1.json")

    # model.summary()
