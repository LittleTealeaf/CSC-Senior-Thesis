import os
from network import Network

network = Network("./data/network")


with open(os.environ["RESULTS_PATH"], "w") as file:
    file.writelines(["id,time"])
