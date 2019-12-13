import matplotlib.pyplot as plt
import numpy as np
from random import random, randint
from scipy.spatial import distance
from data import load_coordinates
from time import time
import os

FILE1 = "TSP-Configurations/eil51.tsp.txt"
FILE2 = "TSP-Configurations/a280.tsp.txt"
ITER = 2000000
bestDistanceList = []

class TSP(object):
    """
    This object contains all the methods to perfrom simulations
    with simulated annealing to solve the Traveling Salesman Problem
    """

    def __init__(self, coords, start_temp, stop_temp, scheme, factor):
        """
        Initialize object with all the coordinates points 
        and a distance matrix, amount of cites, 
        a random initial tour trough the cities and 
        a tracking variable for the distance (initially is inf)
        """

        # Loads introduction information and coordinates cities, 
        # save amount of cities
        self.coords = coords
        self.cities = len(coords)

        # Starting temperature, stop temperature and tracking variable for 
        # current temperature with given cooling schem
        self.start_temp = start_temp
        self.curr_temp = start_temp
        self.stop_temp = stop_temp
        self.scheme = scheme
        self.factor = factor

        # Maximum iterations for simulated annealing. Also initialize a counter 
        # for the amounnt of iterations, of no switches during the annealing 
        # and set a maximum amount of no switches (stationary) 
        self.iterations = 0
        self.no_improvements = 0
        self.stationary = 10000
        self.convergence = False

        # Determinse the distance matrix with at index (i, j) the distance to go
        # from city i to j.
        self.dist_mat = self.dist_matrix()

        # Initiliaze as initial tour a random solution, 
        # a tracking variable for current best tour and for the entire tour 
        # history during the annealing process
        # self.current_tour = self.nearest_neighbour_sol()
        self.current_tour = list(np.random.permutation(self.cities))
        self.best_tour = self.current_tour
        self.tour_history = [self.current_tour]

        # Determine the distance for the current solution (tour) and initialize 
        # a tracking variable for the best distance and the entire distance 
        # history
        self.current_distance = self.calc_distance(self.current_tour)
        self.best_distance = self.current_distance
        self.distance_history = [self.current_distance]
        

        print("Intial distance: ", self.current_distance)

    def cooling_scheme(self):
        """
        """

        if self.scheme == "linear":
            self.curr_temp = self.start_temp - self.factor * self.iterations

        elif self.scheme == "exponential":
            self.curr_temp = self.start_temp * self.factor ** self.iterations


    def dist_matrix(self):
        """
        Returns distance matrix for an array with Euclidean coordinates.
        """
        dist_mat = np.zeros((self.cities, self.cities))
        for i in range(self.cities):
            for j in range(self.cities):
                dist_mat[i, j] = distance.euclidean(self.coords[i], 
                                                    self.coords[j])

        return dist_mat

    def nearest_neighbour_sol(self):
        """
        Determine the nearest neaighbourhood solution for the given
        solution
        """

        city = randint(0, self.cities - 1)
        result = [city]
        cities_to_visit = [x for x in range(self.cities) if x != city]

        while cities_to_visit:
            nearest_city = min([(self.dist_mat[city, j], j) for j in 
                                cities_to_visit], key=lambda x: x[0])
            city = nearest_city[1]
            cities_to_visit.remove(city)
            result.append(city)

        return result

    def calc_distance(self, tour):
        """
        Determines the distance for current tour.
        """

        # Determines the distance from first city until last city.
        distance = 0
        for city in range(self.cities - 1):
            distance += self.dist_mat[tour[city], tour[city + 1]]

        # Also include last connection (from last to first city)
        distance += self.dist_mat[tour[-1], tour[0]]
        return distance

    def accept_prob(self, candidate_distance):
        """
        Determines acceptance probability (according to Metropolis algorithm) 
        of new solution (neighbour).
        """

        return np.exp(-abs(candidate_distance - self.current_distance) / 
                        self.curr_temp)

    def anneal_2opt(self):
        """
        Annealing process with 2-opt
        described here: https://en.wikipedia.org/wiki/2-opt
        """

        tour = self.current_tour.copy()
        begin = randint(1, self.cities - 3)
        end = randint(begin + 1, self.cities - 2)
        tour[begin:end] = list(reversed(tour[begin:end]))
        return tour

    def better_sol(self, dist):
        """
        """
        return dist < self.current_distance

    def best_sol(self, dist):
        """
        """
        return dist < self.best_distance

    def accept(self, tour):
        """
        """
        distance = self.calc_distance(tour) 
        if self.better_sol(distance) and self.best_sol(distance):
            self.current_distance = distance
            self.current_tour = tour
            self.best_distance = distance
            self.best_tour = tour
            self.no_improvements = 0

        elif self.better_sol(distance) and not self.best_sol(distance):
            self.current_distance = distance
            self.current_tour = tour
            self.no_improvements = 0

        elif random() < self.accept_prob(distance):
            self.current_distance = distance
            self.current_tour = tour
            self.no_improvements += 1

        else:
            self.no_improvements += 1

    def is_cooled_down(self):
        
        return self.curr_temp < self.stop_temp

    def is_stationary(self):
        
        return self.no_improvements >= self.stationary

    def finished_annealing(self):
        """
        """
        return self.is_cooled_down() or self.is_stationary()

    def simulated_annealing(self):
        
        while not self.finished_annealing():
            candidate_tour = self.anneal_2opt()
            self.accept(candidate_tour)
            self.iterations += 1
            self.cooling_scheme()
            self.distance_history.append(self.current_distance)
            self.tour_history.append(self.current_tour)
        
        if self.is_stationary():
            self.convergence = True

        print("Minimum distance: ", self.best_distance)
        print("Improvement: ",
              round((self.distance_history[0] - self.best_distance) / \
                    (self.distance_history[0]), 4) * 100, '%')
        print("Iterations: ", self.iterations)
        print("Convergence: ", self.convergence)


    def plot_learning(self, filename):
        plt.plot([i for i in range(len(self.distance_history))], 
                    self.distance_history, label=self.scheme)
        plt.axhline(y=self.distance_history[0], color='r', 
                                linestyle='--', label="Initial distance")
        plt.axhline(y=self.best_distance, color='g', linestyle='--', 
                                label="Best distance")
        plt.axhline(y=426, color="orange", linestyle="--", 
                    label="Optimal distance")
        plt.legend()
        plt.ylabel("Estimated distance")
        plt.xlabel("Iterations")
        plt.title("Learning curve SA")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    coords = load_coordinates(FILE2)
    path = "a280/init_temp"
    temps = []
    with open(f"{path}/init_temps.txt", 'r') as f:
        temps = f.read().splitlines()
        temps = [float(temp) for temp in temps]

    start_temp, stop_temp = temps[-1], 1e-6
    factor_linear = start_temp / ITER
    tsp_linear = TSP(coords, temps[-1], stop_temp, "linear", factor_linear)
    # np.save("eil51/coordinates.npy", tsp_linear.coords)
    # np.save("eil51/distance_matrix.npy", tsp_linear.dist_mat)
    start = time()
    tsp_linear.simulated_annealing()
    # tsp_linear.histogram()
    print("Python scrip ran in {:2.3} minutes.".format((time()-start)/60))

    path = "eil51/linear_cooling"
    # os.makedirs(path, exist_ok=True)
    # dist_history = np.array(tsp_linear.distance_history)
    # tour_history = np.array(tsp_linear.tour_history)
    # np.save(f"{path}/distance_history_1mln.npy", dist_history)
    # np.save(f"{path}/tour_history_1mln.npy", tour_history)
    # tsp_linear.plot_learning(f"{path}/learning_plot.pdf")

    with open(f"{path}/best_solution_1mln.txt", 'w') as f:
        f.write("Distance: " + str(tsp_linear.best_distance) + "\n")
        f.write("\n".join([str(city) for city in tsp_linear.best_tour]))
        # f.write("\n".join([str(city) for city in tsp_linear.best_tour]))

    # factor_exp = np.exp(np.log(stop_temp / start_temp) / ITER)
    # tsp_exp = TSP(coords, start_temp, stop_temp, "exponential", factor_exp)
    # start = time()
    # tsp_exp.simulated_annealing()
    # print("Python scrip ran in {:2.3} minutes.".format((time()-start)/60))

    # path ="eil51/exp_cooling"
    # os.makedirs(path,  exist_ok=True)
    # dist_history = np.array(tsp_exp.distance_history)
    # tour_history = np.array(tsp_exp.tour_history)
    # np.save(f"{path}/distance_history_1mln.npy", dist_history)
    # np.save(f"{path}/tour_history_1mln.npy", tour_history)
    # tsp_exp.plot_learning(f"{path}/learning_plot.pdf")

    # with open(f"{path}/best_solution_1mln.txt", 'w') as f:
    #     f.write("Distance: " + str(tsp_exp.best_distance) + "\n")
    #     f.write("\n".join([str(city) for city in tsp_exp.best_tour]))
