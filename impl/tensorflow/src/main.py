# TODO: Copy over other file. No more classes, we just use @tf.function and functions now
import os
import numpy as np
import tensorflow as tf
import keras
import time

PROJECT_ROOT = "." if "PROJECT_ROOT" in os.environ else "../../.."


def string_to_tensor(string: str):
    "Converts a comma separated string of variables into a Tensor"
    return tf.convert_to_tensor(
        np.fromstring(string, dtype=np.float64, sep=","), dtype=tf.float64
    )


def load_network(file_name: str):
    "Loads the network from a given file"
    layers = []
    with open(file_name) as file:
        sections = file.read().split("\n\n")

        for i, section in enumerate(sections):
            lines = section.splitlines()

            if len(lines) == 0:
                continue

            lines.pop(0)
            biases = tf.Variable(string_to_tensor(lines.pop(0)), name=f"{i}-bias")
            weights = tf.Variable(
                [string_to_tensor(string) for string in lines],
                name=f"{i}-weights",
            )

            layers.append((weights, biases))
    return layers


@tf.function
def back_propogate(inputs, expected, layers, optimizer):
    trainable_variables = []
    with tf.GradientTape() as tape:
        variables = inputs
        for weights, biases in layers:
            variables = variables @ weights
            variables = variables + biases
            variables = keras.activations.relu(variables)
            trainable_variables.extend([weights, biases])
        loss_raw = tf.reshape(variables, (len(inputs),)) - expected
        loss_sqr = tf.math.square(loss_raw)
        loss_mean = tf.reduce_mean(loss_sqr)
        gradient = tape.gradient(loss_mean, trainable_variables)

        optimizer.apply_gradients(zip(gradient, trainable_variables))


##############################################################

print("Load Data")
with open(f"{PROJECT_ROOT}/data/data.csv") as file:
    lines = file.readlines()
    DATA = []
    for line in lines:
        line_data = np.fromstring(line, dtype=np.float64, sep=",")
        expected = np.array(line_data[0], dtype=np.float64)
        variables = np.array(line_data[1:], dtype=np.float64)
        DATA.append((variables, expected))

print("Load Bootstraps")
with open(f"{PROJECT_ROOT}/data/bootstraps.csv") as file:
    lines = file.readlines()
    BOOTSTRAPS = [[int(i) for i in line.split(",")] for line in lines]


print("Create Network")
layers = load_network(f"{PROJECT_ROOT}/data/network")

times = []

optimizer = keras.optimizers.SGD(learning_rate=0.1)

print("Starting Training")
for i, bootstrap in enumerate(BOOTSTRAPS):
    inputs = np.array([DATA[i][0] for i in bootstrap])
    expected = np.array([DATA[i][1] for i in bootstrap])

    inputs = tf.constant(inputs)
    expected = tf.constant(expected)

    start = time.time_ns()

    back_propogate(inputs, expected, layers, optimizer)

    end = time.time_ns()

    elapsed = end - start
    times.append(elapsed)

with open(f"{PROJECT_ROOT}/impl/tensorflow/results.csv", "w") as file:
    data = ["id,time", *[f"\n{index},{elapsed}" for index, elapsed in enumerate(times)]]
    file.writelines(data)
