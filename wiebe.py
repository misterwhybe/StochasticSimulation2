import random
import simpy
import numpy as np
import time

RANDOM_SEED = 42
CUSTOMERS = 2500
INTERVAL_COSTUMERS = 1.0
SERVING_TIME = 0.9
ITER = 300
LESS_ITER = 100
SERVING_TIMES = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                    0.98, 0.99, 0.999]


def source_MMn(env, interval, server, serving_time, iterations, servers, 
                diff, data):
    """
    Source generates customers randomly
    """

    for i in range(CUSTOMERS):
        person = costumer_shortest_job(env, i, server, serving_time, iterations, servers, 
                            diff, data)
        env.process(person)
        arrival_time = random.expovariate(1.0 / interval)
        yield env.timeout(arrival_time)


def costumer(env, name, server, serving_time, iterations, servers, diff, data):
    """
    Costumer arrives, is served and leaves.
    """

    arrive = env.now

    with server.request() as req:
        yield req

        # We got to the counter
        wait = env.now - arrive
        t = random.expovariate(1.0 / serving_time)
        response = wait + t
        yield env.timeout(t)
        leave = env.now
        data[diff, iterations, name, :] = np.array([arrive, wait, response,
                                                        leave])
def costumer_shortest_job(env, name, server, serving_time, iterations, servers, diff, data):
    """
    Costumer arrives, is served and leaves.
    """

    arrive = env.now
    t = random.expovariate(1.0 / serving_time)

    with server.request(priority=t) as req:
        yield req

        # We got to the counter
        wait = env.now - arrive
        
        response = wait + t
        yield env.timeout(t)
        leave = env.now
        data[diff, iterations, name, :] = np.array([arrive, wait, response,
                                                        leave])
def run_sim(iterations, servers, serving_time, interval, diff, data):
    """
    Performs one simulation for a given amount of servers 
    in the facility.
    """

    # random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Start processes and run simulation
    server = simpy.Resource(env, capacity=servers)

    # add procces to environment
    arrival_rate = interval / servers
    env.process(source_MMn(env, arrival_rate, server, serving_time,
                           iterations, servers, diff, data))

    # run simulations
    env.run()

    # return data
def run_sim_shortest_job(iterations, servers, serving_time, interval, diff, data):
    """
    Performs one simulation for a given amount of servers 
    in the facility.
    """

    # random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Start processes and run simulation
    server = simpy.PriorityResource(env, capacity=servers)

    # add procces to environment
    arrival_rate = interval / servers
    env.process(source_MMn(env, arrival_rate, server, serving_time,
                           iterations, servers, diff, data))

    # run simulations
    env.run()

    # return data

def sim_MMn(total_servers, serving_time, interval):

    # Setup and start the simulation
    print("DES")

    diff_servers  = list(range(len(total_servers)))
    data = np.zeros((len(total_servers), ITER, CUSTOMERS, 4))
    for servers, diff in zip(total_servers, diff_servers):
        for i in range(ITER):
            run_sim_shortest_job(i, servers, serving_time, interval, diff, data)
    return data

def sim_diff_rho(total_servers, interval):

    print("DES")
    # serving_times = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 
    #                     0.98, 0.99, 0.999]
    data_rhos = np.zeros((len(SERVING_TIMES), len(total_servers), LESS_ITER, 
                            CUSTOMERS, 4))
    diff_rhos = list(range(len(SERVING_TIMES)))
    diff_servers = list(range(len(total_servers)))

    for serving_time, index_rho in zip(SERVING_TIMES, diff_rhos):
        data = np.zeros((len(total_servers), LESS_ITER, CUSTOMERS, 4))

        for servers, index_server in zip(total_servers, diff_servers):
            for i in range(LESS_ITER):
                run_sim_shortest_job(i, servers, serving_time, interval, index_server, data)

        data_rhos[index_rho, :, :, :, :] = data
    
    return data_rhos


if __name__ == "__main__":

    total_servers = [1, 2, 4]
    # start = time.time()
    # data = sim_MMn(total_servers, SERVING_TIME, INTERVAL_COSTUMERS)
    # print("Python script ran in {:2.3} minutes."
    #       .format((time.time() - start) / 60))

    # np.save("results/data_startup.npy", data)
    # working?

    start = time.time()
    data_rhos = sim_diff_rho(total_servers, INTERVAL_COSTUMERS)
    print("Python script ran in {:2.3} minutes."
          .format((time.time() - start) / 60))
    np.save("results/data_diff_rhos.npy", data_rhos)
