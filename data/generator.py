from random import Random

random = Random()

VARIABLES = 10
OBSERVATIONS = 1000
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SIZE = 1000

LAYERS = [20, 10, 10]


# Generate Data
def generate_row():
    return [random.random() for _ in range(VARIABLES)]


data = [generate_row() for _ in range(OBSERVATIONS)]

with open("data.csv", "w") as file:
    file.write("\n".join([",".join([str(i) for i in row]) for row in data]))


def generate_bootstrap():
    return random.choices(range(OBSERVATIONS), k=BOOTSTRAP_SIZE)


# Generate Bootstraps

bootstraps = [generate_bootstrap() for _ in range(BOOTSTRAP_SAMPLES)]

with open("bootstraps.csv", "w") as file:
    file.write("\n".join([",".join([str(i) for i in row]) for row in bootstraps]))


# Generate Neural Network

LAYERS.append(1)

layer_sizes = [i for i in LAYERS]
layer_sizes.insert(0, VARIABLES)


with open("network", "w") as file:
    for i in range(len(LAYERS)):
        size_in = layer_sizes[i]
        size_out = layer_sizes[i + 1]

        file.write(f"{size_in} {size_out}\n")
        file.write(",".join([str(random.random() - 0.5) for _ in range(size_out)]))
        file.write("\n")
        file.write(
            "\n".join(
                [
                    ",".join([str(random.random() - 0.5) for _ in range(size_out)])
                    for _ in range(size_in)
                ]
            )
        )
        file.write("\n\n")