from random import Random

import shutil
import os
import tomllib

random = Random()

with open("data.toml", "rb") as file_bootstraps:
    config = tomllib.load(file_bootstraps)


os.mkdir(os.path.join("out", "data", "vars"))

# TODO: Change the ranges to variables

print("Generating Index")
with open(os.path.join("out", "data", "index"), "w") as var_index:
    var_index.write(
        ",".join(
            [
                str(i)
                for i in range(
                    config["variables"]["min"],
                    config["variables"]["max"],
                    config["variables"]["step"],
                )
            ]
        )
    )
    var_index.write("\n")
    var_index.write(
        ",".join(
            [
                str(i)
                for i in range(
                    config["bootstrap"]["min"],
                    config["bootstrap"]["max"],
                    config["bootstrap"]["step"],
                )
            ]
        )
    )


print("Generating Bootstraps")


def generate_bootstrap():
    return random.choices(range(config["observations"]), k=config["bootstrap"]["max"])


bootstraps = [generate_bootstrap() for _ in range(config["samples"])]
with open(os.path.join("out", "data", "bootstraps"), "w") as file_bootstraps:
    file_bootstraps.write(
        "\n".join([",".join([str(i) for i in row]) for row in bootstraps])
    )


print("Generating Data")


for VARIABLES in range(
    config["variables"]["min"],
    config["variables"]["max"],
    config["variables"]["step"],
):
    os.mkdir(os.path.join("out", "data", "vars", str(VARIABLES)))
    print(f"{VARIABLES} \t Generating Data")

    def generate_row():
        # First one is the expected output
        return [random.random() for _ in range(VARIABLES + 1)]

    data = [generate_row() for _ in range(config["observations"])]
    with open(
        os.path.join("out", "data", "vars", str(VARIABLES), "data"), "w"
    ) as file_data:
        file_data.write("\n".join([",".join([str(i) for i in row]) for row in data]))
    # Generate Neural Network
    print(f"{VARIABLES}\t Generating Neural Network")
    layer_sizes = [int(VARIABLES * i) for i in config["layers"]]
    layer_sizes.append(1)
    layer_sizes.insert(0, VARIABLES)
    with open(
        os.path.join("out", "data", "vars", str(VARIABLES), "network"), "w"
    ) as file:
        for i in range(len(config["layers"]) + 1):
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
            if i < len(config["layers"]):
                file.write("\n\n")



# print(config['variables'])


# VARIABLES = config['variables']
# OBSERVATIONS = config['observations']
# BOOTSTRAP_SAMPLES = config['samples']
# BOOTSTRAP_SIZE = config['bootstrap']
# LAYERS = config['layers']

# if os.path.exists("data"):
#     shutil.rmtree("data")
# os.mkdir("data")

# # Generate Data
# print("Generating Data")


# with open("data/data.csv", "w") as file:
#     file.write("\n".join([",".join([str(i) for i in row]) for row in data]))


# # Generate Bootstraps
# print("Generating Bootstraps")
