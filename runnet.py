import sys
import numpy as np
from read_file import read_wnet, read_file


def runNet(network, inputs, labels):
    predictions = network.forward(inputs)
    accuracy = np.mean(predictions == labels)
    return accuracy


def main():
    wnet_file = sys.argv[1]
    network = read_wnet(wnet_file)
    test_file = sys.argv[2]
    inputs, labels = read_file(test_file)
    accuracy = runNet(network, inputs, labels)
    print("Accuracy: ", accuracy)


if __name__ == '__main__':
    main()
