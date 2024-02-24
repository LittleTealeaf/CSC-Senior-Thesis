import os
import shutil
import numpy as np
import time



PROJECT_ROOT = "." if "PROJECT_ROOT" in os.environ else "../.."

OUT_PATH = os.environ["OUT_PATH"] if "OUT_PATH" in os.environ else None

if OUT_PATH is not None:
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_PATH, exist_ok=True)

def string_to_array(string: str):
    return np.fromstring(string, dtype=np.float64, sep=",")

def load_network(file_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    with open(file_name) as file:
        sections = file.read().split("\n\n")
        for i, section in enumerate(sections):
            lines = section.splitlines()
            if len(lines) == 0:
                continue
            lines.pop(0)
            biases = string_to_array(lines.pop(0))
            weights = np.array([string_to_array(string) for string in lines])
            layers.append((weights, biases))
    return layers

def forward_pass(inputs, layers):
    variables = inputs
    for weights, biases in layers:
        variables = np.dot(variables, weights) + biases
        variables = np.maximum(variables, 0)  # ReLU activation
    return variables

def back_propagate(inputs, expected, layers):
    trainable_variables = []  # Stores weights and biases

    # Forward Pass
    activations = [inputs]
    variables = inputs
    for weights, biases in layers:
        variables = np.dot(variables, weights) + biases
        variables = np.maximum(variables, 0)  # ReLU
        activations.append(variables)
    trainable_variables.extend(layers)

    # Error and Gradient Calculation
    loss_raw = variables - expected
    loss_sqr = loss_raw ** 2
    loss_mean = np.mean(loss_sqr)

    gradients = []
    error = 2 * loss_raw  # Gradient of squared error

    for i in reversed(range(len(layers))):
        dW = np.dot(activations[i].T, error)
        db = np.sum(error, axis=0)
        gradients.insert(0, (dW, db))

        error = np.dot(error, layers[i][0].T)
        error[activations[i] <= 0] = 0  # ReLU gradient

    return gradients, loss_mean

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


layers = load_network(f"{PROJECT_ROOT}/data/network")

optimizer_lr = 0.1  # Learning rate

times = []

print("Starting Training")
for i, bootstrap in enumerate(BOOTSTRAPS):
    inputs = np.array([DATA[i][0] for i in bootstrap])
    expected = np.array([DATA[i][1] for i in bootstrap])

    start = time.time_ns()

    gradients, loss = back_propagate(inputs, expected, layers)

    # Update weights and biases
    for (weights, biases), (dW, db) in zip(layers, gradients):
        weights -= optimizer_lr * dW
        biases -= optimizer_lr * db

    end = time.time_ns()
    times.append(end - start)

if OUT_PATH is not None:
    with open(f"{OUT_PATH}/times.csv", "w") as file:
        data = [
            "id,time",
            *[f"\n{index},{elapsed}" for index, elapsed in enumerate(times)],
        ]
        file.writelines(data)

    with open(f"{OUT_PATH}/network", "w") as file:
        data = []
        for weights, biases in layers:
            (height, width) = weights.shape
            data.append(f"{height} {width}\n")
            data.append(",".join([str(i) for i in biases.numpy()]))
            data.append("\n")
            data.append("\n".join([",".join([str(i) for i in row.numpy()]) for row in weights]))
            data.append("\n\n")
        file.writelines(data)
