import numpy as np
import random
import math
import matplotlib.pyplot as plt


from src.utils import read_csv, random_step_single, square, f


def mse(weights, input, nodes_number, stride):
    """
    Computes the mean squared error of the predicted sequence and the true sequence

    """
    ret = 0
    for i in range(0, len(input) - nodes_number, stride):
        ret += sum(square(np.subtract(f(weights, input[i:i + nodes_number]), input[i + 1:i + nodes_number + 1])))
    return ret


def annealing(start_weights, start_energy, T, input, nodes_number, stride):
    """
    Performs the change of state by shuffling the weights in the cognitive map matrix

    """
    new_weights = np.vectorize(random_step_single)(start_weights)
    new_energy = mse(new_weights, input, nodes_number, stride)
    diff = new_energy - start_energy
    if diff <= 0 or random.uniform(0, 1) < math.exp(-diff / T):
        return new_weights, new_energy
    else:
        return start_weights, start_energy


def decrease_temp(T, cool_parameter):
    return T * cool_parameter


def train(_start_temp, _end_temp, _eq_number, _cool_number, nodes_number, stride):
    """
    Trains the model by applying the optimization by simulated annealing

    """
    input = read_csv(nodes_number, stride)
    weights = np.zeros((nodes_number, nodes_number))

    start_temp = _start_temp
    end_temp = _end_temp
    T = start_temp
    energy = mse(weights, input, nodes_number, stride)
    eq_number = _eq_number
    cool_parameter = _cool_number

    energies = [energy]
    best_energies = [energy]

    best_weights = weights
    best_energy = energy

    while T >= end_temp:
        print(T)
        # stay on the same temp (in the equilibrium) for eq_number iterations
        for _ in range(eq_number):
            weights, energy = annealing(weights, energy, T, input, nodes_number, stride)
            T = decrease_temp(T, cool_parameter)

            if energy < best_energy:
                best_energy = energy
                best_weights = weights

            energies.append(energy)
            best_energies.append(best_energy)

    plt.plot(energies, label="Energy")
    plt.plot(best_energies, label="Best energy")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    return best_energy


def get_parameters():
    """
    Gets the simulated annealing parameters from the user

    """
    print("================")
    start_temp = end_temp = eq_number = cool_number = nodes_number = stride = 0

    print("Simulated annealing parameters\n")
    while start_temp <= 0:
        start_temp = float(input("Start temperature (ts: ts > 0): "))
    while 0 >= end_temp or end_temp > start_temp:
        end_temp = float(input("End temperature (te: te > 0 and te < ts): "))
    while eq_number <= 0:
        eq_number = int(input("Equilibrium number (e: e >= 1 and e is an integer): "))
    while cool_number <= 0 or cool_number > 1:
        cool_number = float(input("Cooling parameter (c: 0 < c < 1): "))
    while nodes_number <= 0:
        nodes_number = int(input("Number of nodes (n: n >= 1 and n is an integer): "))
    while stride <= 0:
        stride = int(input("Stride (s: s >= 1): "))

    return start_temp, end_temp, eq_number, cool_number, nodes_number, stride


if __name__ == '__main__':
    params = get_parameters()
    results = train(*params)
    print("Mean squared error: ", results)