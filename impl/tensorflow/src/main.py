import os
import numpy as np
import tensorflow as tf
import keras
import time


def convert_string_to_tensor(string: str, name: str):

    np_array = np.fromstring(string, dtype=np.float64, sep=",")

    return tf.Variable(tf.convert_to_tensor(np_array, dtype=tf.float64), name=name)


@tf.function
def feed_forward_tf(inputs, layers):
    print("Trace Feed Forward")
    variables = inputs
    for weights, biases in layers:
        variables = variables @ weights
        variables = variables + biases
        variables = keras.activations.relu(variables)
    return variables


@tf.function
def back_prop_loss_tf(inputs, expected, layers):
    print("Trace Back Propagation Loss")
    variables = feed_forward_tf(inputs, layers)
    loss = tf.reshape(variables, (len(inputs),)) - expected
    loss = tf.math.square(loss)
    return tf.reduce_mean(loss)

class Network:
    def __init__(self, file_name: str) -> None:
        self.layers = []
        self.trainable_variables = []

        # Loads data from the file into the layers and train
        with open(file_name) as file:
            sections = file.read().split("\n\n")

            for i, section in enumerate(sections):
                lines = section.splitlines()

                if len(lines) == 0:
                    continue

                lines.pop(0)
                biases = convert_string_to_tensor(lines.pop(0), f"{i}-bias")
                weights = [
                    convert_string_to_tensor(string, f"{i}-{j}-weight")
                    for j, string in enumerate(lines)
                ]
                self.trainable_variables.append(biases)
                self.trainable_variables.extend(weights)
                self.layers.append((weights, biases))

    def back_propagate(self, inputs, expected_outputs, alpha):

        with tf.GradientTape() as tape:

            loss = back_prop_loss_tf(inputs, expected_outputs, self.layers)

            grad = tape.gradient(loss, self.trainable_variables)

            optimizer = keras.optimizers.SGD(learning_rate=alpha)

            optimizer.apply_gradients(zip(grad, self.trainable_variables))


#################################################################

RELATIVE = "./" if "PROJECT_ROOT" in os.environ else "../../../"

print("Data: Begin Loading")
with open(f"{RELATIVE}data/data.csv") as file:
    lines = file.readlines()
    DATA = []
    for line in lines:
        line_data = np.fromstring(line, dtype=np.float64, sep=",")
        expected = np.array(line_data[0], dtype=np.float64)
        variables = np.array(line_data[1:], dtype=np.float64)
        DATA.append((variables, expected))
print("Data: Loaded")

print("Bootstraps: Begin Loading")
with open(f"{RELATIVE}data/bootstraps.csv") as file:
    lines = file.readlines()
    BOOTSTRAPS = [[int(i) for i in line.split(",")] for line in lines]
print("Bootstraps: Loaded")


print("Network: Begin Creating")
network = Network(f"{RELATIVE}data/network")
print("Network: Created")


times = []

for i, bootstrap in enumerate(BOOTSTRAPS):
    inputs = np.array([DATA[i][0] for i in bootstrap])
    expected = np.array([DATA[i][1] for i in bootstrap])

    inputs = tf.constant(inputs)

    print(f"Iter {i}")

    start = time.time_ns()

    network.back_propagate(inputs, expected, 0.1)

    end = time.time_ns()

    elapsed = end - start

    times.append(elapsed)


with open(os.environ["RESULTS_PATH"], "w") as file:
    data = ["id,time", *[f"\n{index},{elapsed}" for index, elapsed in enumerate(times)]]
    file.writelines(data)
