import json
import os
import numpy as np
from neural_network import NeuralNetwork


def read_file(fileName):
    inputs = []
    labels = []
    with open(fileName, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                txt_input, txt_label = line.split('   ')
                input = one_hot_encode(txt_input)
                label = int(txt_label)
                inputs.append(input)
                labels.append(label)
    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels


def one_hot_encode(string):
    string_array = list(string)
    string_array_len = len(string_array)
    input_array = []
    for i in range(string_array_len):
        input_array.append(int(string_array[i]))
    return input_array


def write_wnet(fileName, nn):
    if os.path.exists(fileName):
        os.remove(fileName)
    with open(fileName, 'w') as file:
        nn_items = vars(nn).items()
        nn_dict = {str(k): v.tolist() if isinstance(v, np.ndarray) else v for k, v in nn_items}
        json_string = json.dumps(nn_dict)
        file.write(json_string)

def read_wnet(fileName):
    network = None
    with open(fileName, 'r') as file:
        for line in file:
            json_string = line.strip()  # Remove leading/trailing whitespace
            if line:
                json_dict = json.loads(json_string, object_hook=numpy_decoder)
                network = NeuralNetwork(input_size=json_dict['input_size'], hidden_size=json_dict['hidden_size'])
                network.w1 = json_dict['w1']
                network.w2 = json_dict['w2']
                network.b1 = json_dict['b1']
                network.b2 = json_dict['b2']
    return network


def numpy_decoder(obj):
    if '__ndarray__' in obj:
        # Reconstruct NumPy ndarray from the 'data' key
        return np.array(obj['data'])
    return obj

