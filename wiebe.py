import matplotlib.pyplot as plt
import numpy as np
from random import random, randint
from scipy.spatial import distance
from data import load_coordinates
from time import time
import os

from tsp import TSP
FILE1 = "TSP-Configurations/eil51.tsp.txt"
ITER = 2000000
# def inputs():
#     start_temp, stop_temp = temps[-1], 1e-6
#     factor_linear = start_temp / ITER
    # tsp_linear = TSP(coords, temps[-1], stop_temp, "linear", factor_linear)
    # tsp_linear.simulated_annealing()


def histogram():
    num_bins = 40
    start_temp, stop_temp = temps[-1], 1e-6
    bestDistanceListEX = []
    bestDistanceListL = []
    factor_linear = start_temp / ITER
    factor_exp = np.exp(np.log(stop_temp / start_temp) / ITER)
    for i in range(200):
        tsp_linear = TSP(coords, temps[-1], stop_temp, "linear", factor_linear)
        tsp_exp = TSP(coords, start_temp, stop_temp, "exponential", factor_exp)
        tsp_linear.simulated_annealing()
        tsp_exp.simulated_annealing()
        print("Linear:", tsp_linear.best_distance)
        print("Exponential:", tsp_exp.best_distance)
        bestDistanceListL.append(round(tsp_linear.best_distance,2))
        bestDistanceListEX.append(round(tsp_exp.best_distance,2))
        print("iteration:", i)
    print(bestDistanceListL)
    print(bestDistanceListEX)
    # plt.hist(bestDistanceListL, num_bins,facecolor='blue', alpha=0.5)
    # plt.xlabel('Distance')
    # plt.ylabel('Amount')
    # plt.title('Best estimated distances for linear cooling schemes')
    # plt.savefig("eil51/linear_cooling/histogram_linear.pdf", dpi=300)
    # newbestDistanceListL = np.asarray(bestDistanceListL)
    # np.save(f"eil51/LinearBestDistanceHist.npy", newbestDistanceListL)

    # plt.show()
    # plt.hist(bestDistanceListEX,num_bins, facecolor='blue', alpha=0.5)
    # plt.xlabel('Distance')
    # plt.ylabel('Amount')
    # plt.title('Best estimated distances for exponential cooling schemes')
    # plt.savefig("eil51/exp_cooling/histogram_exponential.pdf", dpi=300)
    # newbestDistanceListEX = np.asarray(bestDistanceListEX)
    # np.save(f"eil51/EXBestDistanceHist.npy", newbestDistanceListEX)

    plt.show()
    return bestDistanceListL, bestDistanceListEX


if __name__ == "__main__":
    coords = load_coordinates(FILE1)
    path = "eil51/init_temp"
    temps = []
    with open(f"{path}/init_temps.txt", 'r') as f:
        temps = f.read().splitlines()
        temps = [float(temp) for temp in temps]

    # start_temp, stop_temp = temps[-1], 1e-6
    # factor_linear = start_temp / ITER
    # tsp_linear = TSP(coords, temps[-1], stop_temp, "linear", factor_linear)
    # np.save("eil51/coordinates.npy", tsp_linear.coords)
    # np.save("eil51/distance_matrix.npy", tsp_linear.dist_mat)
    start = time()
    histogram()
    # tsp_linear.simulated_annealing()
    # tsp_linear.histogram()
    print("Python scrip ran in {:2.3} minutes.".format((time()-start)/60))

    # path = "eil51/linear_cooling"
    # os.makedirs(path, exist_ok=True)
    # dist_history = np.array(tsp_linear.distance_history)
    # tour_history = np.array(tsp_linear.tour_history)
    # np.save(f"{path}/distance_history_1mln.npy", dist_history)
    # np.save(f"{path}/tour_history_1mln.npy", tour_history)
    # tsp_linear.plot_learning(f"{path}/learning_plot.pdf")

    # with open(f"{path}/best_solution_1mln.txt", 'w') as f:
    #     f.write("Distance: " + str(tsp_linear.best_distance) + "\n")
    #     f.write("\n".join([str(city) for city in tsp_linear.best_tour]))
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
