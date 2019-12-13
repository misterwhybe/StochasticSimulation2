import numpy as np
from scipy.spatial import distance
from random import randint, random, uniform
from tqdm import tqdm
from time import time
import os

FILE1 = "TSP-Configurations/eil51.tsp.txt"
FILE1_OPT = "TSP-Configurations/eil51.opt.tour.txt"

def load_coordinates(filename):
    """
    Loads coordinates cites for a given file. 
    Returns array: containging coordinates (numpy)
    """

    # Get the amount of cities from the filename
    cities = int(''.join([x for x in filename if x.isdigit()]))

    # Get introductory information and coordinates from file
    lines_info = 6
    coords = []
    with open(filename, 'r') as f:
        
        # Skip first 6 lines
        for _ in range(lines_info):
            f.readline()

        for _ in range(cities):
            line = f.readline().rstrip("\n").split()
            x, y, = float(line[1]), float(line[2])
            coords.append((x, y))

        return np.array(coords)


def dist_matrix(coords):
    """
    Returns distance matrix for an array with Euclidean coordinates.
    """
    cities = len(coords)
    dist_mat = np.zeros((cities, cities))
    for i in range(cities):
        for j in range(cities):
            dist_mat[i, j] = distance.euclidean(coords[i],
                                                coords[j])

    return dist_mat

def load_opt_tour(filename):

    # Get the amount of cities from the filename
    cities = int(''.join([x for x in filename if x.isdigit()]))

    # Get introductory information and coordinates from file
    lines_info = 5
    tour = []
    with open(filename, 'r') as f:

        # Skip first 5 lines
        for _ in range(lines_info):
            f.readline()

        for _ in range(cities):
            tour.append(int(f.readline().rstrip("\n")) - 1)

    return tour


def calc_distance(dist_mat, tour):
    """
    Determines the distance for current tour.
    """

    # Determines the distance from first city until last city.
    distance = 0
    cities = len(tour)
    for city in range(cities - 1):
        distance += dist_mat[tour[city], tour[city + 1]]

    # Also include last connection (from last to first city)
    distance += dist_mat[tour[-1], tour[0]]
    return distance

def anneal_2opt(tour):
    """
    Annealing process with 2-opt
    described here: https://en.wikipedia.org/wiki/2-opt
    """
    candidate_tour = tour.copy()
    cities = len(tour)
    begin = randint(1, cities - 3)
    end = randint(begin + 1, cities - 2)
    candidate_tour[begin:end] = list(reversed(candidate_tour[begin:end]))
    return candidate_tour

def accept_prob(candidate_distance, current_distance, temp):
    """
    Determines acceptance probability (according to Metropolis algorithm) 
    of new solution (neighbour).
    """

    return np.exp(-abs(candidate_distance - current_distance) / temp)

def find_initial_temp(dist_mat, cities):
    """
    """

    target_ratio, max_ratio, convergence = 0.55, 0.70, False
    N, start_temp, curr_ratio = 1000000, 1250, 0
    temps, accept_ratios = [], []

    while curr_ratio < target_ratio or not convergence:

        accepted, lower_values = 0, 0
        tour = np.random.permutation(cities)
        for _ in tqdm(range(N)):
            tour_dist = calc_distance(dist_mat, tour)
            cand_tour = anneal_2opt(tour)
            cand_dist = calc_distance(dist_mat, cand_tour)

            if cand_dist < tour_dist:
                tour = cand_tour
                continue
            elif random() < accept_prob(cand_dist, tour_dist, start_temp):
                tour = cand_tour
                accepted += 1

            lower_values += 1

        curr_ratio = accepted / lower_values
        accept_ratios.append(curr_ratio)
        temps.append(start_temp)

        if curr_ratio >= max_ratio:
            start_temp /= 3
        elif curr_ratio < target_ratio:
            start_temp *= 2 
        else:
            convergence = True

    return temps, accept_ratios
        

if __name__ == "__main__":

    coords = load_coordinates(FILE1)
    dist_mat = dist_matrix(coords)

    opt_tour = load_opt_tour(FILE1_OPT)
    dist_opt = calc_distance(dist_mat, opt_tour)
    os.makedirs("eil51", exist_ok=True)
    with open("eil51/optimal_dist.txt", 'w') as f:
        f.write(str(dist_opt))
        
    temps, accept_ratios = find_initial_temp(dist_mat, len(coords))
    path = "eil51/init_temp"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/init_temps.txt", 'w') as f:
        f.write("\n".join([str(temp) for temp in temps]))

    with open(f"{path}/init_temp_accept_ratios.txt", 'w') as f:
        f.write("\n".join([str(ratio) for ratio in accept_ratios]))
