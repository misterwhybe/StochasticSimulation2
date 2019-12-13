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


def find_initial_temp(dist_mat, cities, chain_length):
    """
    """

    target_ratio, max_ratio, convergence = 0.55, 0.70, False
    N, start_temp, curr_ratio = 10000, 1250, 0
    temps, accept_ratios = [], []

    while curr_ratio < target_ratio or not convergence:

        accepted, lower_values = 0, 0
        tour = np.random.permutation(cities)
        for _ in tqdm(range(N)):
            
            for _ in range(chain_length):
                tour_dist = calc_distance(tour)
                candidate_tour = anneal_2opt(tour)
                candidate_dist = calc_distance(dist_mat, candidate_tour)

                if candidate_dist < tour_dist:
                    tour = candidate_tour
                    continue
                elif random() < accept_prob(candidate_dist, tour_dist, temp):
                    tour = candidate_tour
                    accepted += 1

                lower_values += 1

            curr_ratio = 