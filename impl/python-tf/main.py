import os
import shutil
import numpy as np
import tensorflow as tf
import keras
import time


OUT_PATH = os.environ["OUT_PATH"] if "OUT_PATH" in os.environ else None
assert OUT_PATH is not None
DATASET_PATH = os.environ["DATASET_PATH"] if "DATASET_PATH" in os.environ else None
assert DATASET_PATH is not None
NETWORK_PATH = os.environ["NETWORK_PATH"] if "NETWORK_PATH" in os.environ else None
assert NETWORK_PATH is not None
BOOTSTRAP_COUNT = int(os.environ["BOOTSTRAP_COUNT"]) if "BOOTSTRAP_COUNT" in os.environ else None
assert BOOTSTRAP_COUNT is not None
BOOTSTRAP_PATH = os.environ['BOOTSTRAP_PATH'] if 'BOOTSTRAP_PATH' in os.environ else None
assert BOOTSTRAP_PATH is not None


def string_to_tensor(string: str):
    "Converts a comma separated string of variables into a Tensor"
    return tf.convert_to_tensor(
        np.fromstring(string, dtype=np.float64, sep=","), dtype=tf.float64
    )


def load_network(file_name: str) -> list[tuple[tf.Variable, tf.Variable]]:
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

print("Load Bootstraps")
with open(BOOTSTRAP_PATH) as file:
    lines = file.readlines()
    BOOTSTRAPS = [[int(i) for i in line.split(",")] for line in lines]


print("Load Data")
with open(DATASET_PATH) as file:
    lines = file.readlines()
    DATA = []
    for line in lines:
        line_data = np.fromstring(line, dtype=np.float64, sep=",")
        expected = np.array(line_data[0], dtype=np.float64)
        variables = np.array(line_data[1:], dtype=np.float64)
        DATA.append((variables, expected))

print("Create Network")
layers = load_network(NETWORK_PATH)


times = []


print("Starting Training")
optimizer = keras.optimizers.SGD(learning_rate=0.1)
for i, bootstrap in enumerate(BOOTSTRAPS):
    bootstrap = bootstrap[:BOOTSTRAP_COUNT]
    inputs = np.array([DATA[i][0] for i in bootstrap])
    expected = np.array([DATA[i][1] for i in bootstrap])


    inputs = tf.constant(inputs)
    expected = tf.constant(expected)

    start = time.time_ns()

    back_propogate(inputs, expected, layers, optimizer)

    end = time.time_ns()

    elapsed = end - start
    times.append(elapsed)


with open(OUT_PATH, "w") as file:
    file.write("\n".join(f"{index},{elapsed}" for index, elapsed in enumerate(times)))
