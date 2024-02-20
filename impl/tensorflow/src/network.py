import tensorflow as tf
import numpy as np


def convert_string_to_tensor(string):
    return tf.convert_to_tensor(
        np.array(
            tf.strings.split(tf.convert_to_tensor(string), sep=",").numpy(),
            dtype="float64",
        )
    )


class Network:
    def __init__(self, file_name: str) -> None:
        self.layers = []

        with open(file_name) as file:
            sections = file.read().split("\n\n")

            for section in sections:
                lines = section.splitlines()

                if len(lines) == 0:
                    continue

                lines.pop(0)
                biases = convert_string_to_tensor(lines.pop(0))
                weights = [convert_string_to_tensor(string) for string in lines]
                self.layers.append((weights, biases))
