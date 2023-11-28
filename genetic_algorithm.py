import copy
import random
import numpy as np
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork

TOURNAMENT_SIZE = 5
BEST_KEY_DUPLICATION = 0.1
MUTATION_RATE = 0.2

def showGraph(y1, y2, x):
    plt.plot(x, y1, color='blue', label='avg')
    plt.plot(x, y2, color='red', label='best')
    plt.xlabel('fitness value')
    plt.ylabel('iterations')
    plt.legend()
    plt.show()


class GeneticAlgorithm:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.population = []

    def initialPopulation(self, population_size, hidden_size):
        population = [NeuralNetwork(input_size=self.inputs.shape[1], hidden_size=hidden_size)
                      for _ in range(population_size)]
        self.population = population

    def fitness(self, network):
        predictions = network.forward(self.inputs)
        accuracy = np.mean(predictions == self.labels)
        return accuracy

    def selection(self, fitness_scores):
        tournament_indices = random.sample(range(len(self.population)), TOURNAMENT_SIZE)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        best = max(tournament_scores)
        tournament_index = tournament_scores.index(best)
        real_index = tournament_indices[tournament_index]
        return self.population[real_index]

    def reproduction(self, fitness_scores, best_key):
        population_len = len(self.population)
        next_generation = [best_key for _ in range(int(round(population_len * BEST_KEY_DUPLICATION)))]
        population_left = len(self.population) - len(next_generation)
        for _ in range(population_left):
            parent1 = self.selection(fitness_scores)
            parent2 = self.selection(fitness_scores)
            child = self.crossover(parent1, parent2)
            next_generation.append(child)
        self.mutation(next_generation, MUTATION_RATE)
        return next_generation

    def mutation(self, next_generation, mutation_rate):
        next_generation_len = len(next_generation)
        for i in range(next_generation_len):
            mutated_network = copy.deepcopy(next_generation[i])
            mutated_network.w1 += np.random.randn(*mutated_network.w1.shape) * mutation_rate
            mutated_network.w2 += np.random.randn(*mutated_network.w2.shape) * mutation_rate
            mutated_network.b1 += np.random.randn(*mutated_network.b1.shape) * mutation_rate
            mutated_network.b2 += np.random.randn(*mutated_network.b2.shape) * mutation_rate
            next_generation[i] = mutated_network
        return next_generation

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(input_size=parent1.input_size, hidden_size=parent1.hidden_size)
        # crossover_point_w = np.random.randint(1, parent1.input_size)
        # w = np.concatenate((parent1.w[:crossover_point_w], parent2.w[crossover_point_w:]))
        w1 = (parent1.w1 + parent2.w1) / 2
        w2 = (parent1.w2 + parent2.w2) / 2
        b1 = (parent1.b1 + parent2.b1) / 2
        b2 = (parent1.b2 + parent2.b2) / 2
        # change for child
        child.w1 = w1
        child.w2= w2
        child.b1 = b1
        child.b2 = b2
        return child

    def run(self, generations):
        best_solutions = []
        avg_solutions = []
        max_fitness = -float('inf')
        max_network = None

        for generation in range(generations):
            fitness_scores = [self.fitness(network) for network in self.population]
            # calc avg
            avg = np.mean(fitness_scores)
            avg_solutions.append(avg)
            # calc best
            best_fitness = max(fitness_scores)
            best_index = fitness_scores.index(best_fitness)
            best_network = self.population[best_index]
            best_solutions.append(best_fitness)
            print("Generation: ", generation + 1)
            print("Fitness: ", best_fitness)
            print("*************************************")
            # calc max
            if (best_fitness > max_fitness):
                max_fitness = best_fitness
                max_network = best_network
            next_generation = self.reproduction(fitness_scores, best_network)
            self.population = next_generation
        showGraph(avg_solutions, best_solutions, range(1, len(best_solutions) + 1))
        return max_network
