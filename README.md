# Pattern Recognition using Neural Networks

In this exercise, you are provided with two data files, `nn0.txt` and `nn1.txt`. Each file contains 20,000 binary strings of length 16. After each string, there is a digit (0 or 1) indicating whether the string adheres to a specific pattern. In `nn0.txt`, the patterns are relatively easier to identify, while in `nn1.txt`, they are more challenging.

Your task is to construct neural networks using a genetic algorithm to learn these patterns and predict whether a given string conforms to a pattern. You have the flexibility to decide and define the neural network structure in terms of the number of layers and connections. The same structure can be used for both problems, or you can opt for different structures. Split the data into a training set and a testing set (you can choose the proportions) and aim to build a network that performs well on the testing set. We will evaluate your networks on additional data generated with patterns similar to those in our dataset.

To facilitate testing, you need to create two programs for each network:

1. **Network Construction (buildnet):**
   - Input: Two files - a learning file and a testing file (e.g., you can decide to use 15,000 examples for learning and 5,000 for testing, or choose a different ratio).
   - Output: A file named `wnet` containing the network structure and weights. This program, which utilizes a genetic algorithm with multiple learning cycles, may take some time to run.

2. **Network Execution (runnet):**
   - Input: The output file `wnet` from the network construction program (the format of which you can decide for convenience) containing the network structure and weights. Also, a data file with 20,000 binary strings (similar to the original data files but without classification).
   - Output: A text file with 20,000 lines, each line indicating the corresponding classification (0 or 1) for each string. This program does not involve a genetic algorithm but runs a feedforward network on the given data quickly.

To enable evaluation, you need to prepare both programs for each network. During testing, we will assess the performance of your networks on additional data.

---

Note: While constructing the neural network, you have the freedom to decide on the network structure, genetic algorithm parameters, and data split ratios. Ensure that your programs are well-documented, and you can make use of markdown in this README file for clear explanations.
