from genetic_algorithm import GeneticAlgorithm
from read_file import read_file, write_wnet
import sys


def buildNet(x_train, y_train):
    ga = GeneticAlgorithm(inputs=x_train, labels=y_train)
    ga.initialPopulation(population_size=100, hidden_size=8)
    best_network = ga.run(generations=200)
    return best_network


def main():
    # load train file
    train_file = sys.argv[1]
    wnet_file = sys.argv[2]
    inputs, labels = read_file(train_file)
    network = buildNet(inputs, labels)
    write_wnet(wnet_file, network)


if __name__ == '__main__':
    main()

