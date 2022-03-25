from operator import mod
from src.ANN import *
from src.dump import *
import numpy as np
import json

if __name__ == "__main__":
    input_single = np.array([[1, 1]])
    # input_single2 = np.array([[1, 1]])

    # input_batch = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    # output_batch = np.array([[1], [1], [0], [0]])

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
    lines = []
    with open("data_banknote_authentication.txt", "r") as f:
        lines = f.readlines()

    attributes = []
    classes = []

    for line in lines:
        data = line.split(",")
        classes.append([data.pop()])
        attributes.append(data)

    inputdata = np.array(attributes).astype(float)
    outputdata = np.array(classes).astype(float)

    # print(inputdata)
    # print(outputdata)

    inputsingle_0 = np.array([[3.6216, 8.6661, -2.8073, -0.44699]])
    inputsingle_1 = np.array([[-1.7279, -6.841, 8.9494, 0.68058]])

    # # Model RelU + Linear
    model = Sequential(random_state=69)
    model.add(Dense(units=2, input_dim=4, activation_function="relu"))
    model.add(Dense(units=1, activation_function="sigmoid"))
    model.compile("sum_squared_error", 0.0001, 0.01)
    model.fit(inputdata, outputdata, batch_size=10)
    print(model.predict(inputsingle_0))
    print(model.predict(inputsingle_1))
    # model.summary()
    # print(model.predict(input_single))
    # print(model.predict(input_single2))

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
