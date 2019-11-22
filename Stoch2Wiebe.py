
import random

import simpy


RANDOM_SEED = 20
NUM_servers = 2  # Number of parallel servers
helpingTime = 5      # Minutes it takes to clean a car
T_INTER = 3       # Create a customer every ~7 minutes
SIM_TIME = 20     # Simulation time in minutes
num_customers = 200


class DES(object):
    """Our DES has a limited number of servers (``NUM_servers``) to
    help customers in parallel.

    Customers have to request one of the servers. When they got one, they
    can start the helping processes and wait for it to finish.

    """
    def __init__(self, env, num_servers, servingTime):
        self.env = env
        self.server = simpy.Resource(env, NUM_servers)
        self.servingTime = servingTime

    def handleTime(self):
        """The washing processes. It takes a server processes and tries
        to clean it."""
        for i in range(1, num_customers +1):
            # helps for lambda
            helpingTime = random.expovariate(1 / 10)
        yield self.env.timeout(helpingTime)
        print("Took %d minutes." % helpingTime)


def Customer(env, name, cw):
    """The car process (each car has a ``name``) arrives at the carwash
    (``cw``) and requests a cleaning machine.

    It then starts the washing process, waits for it to finish and
    leaves to never come back ...

    """
    print('%s arrives at the server at %.2f.' % (name, env.now))
    with cw.server.request() as request:
        yield request

        print('%s enters the server at %.2f.' % (name, env.now))
        yield env.process(cw.handleTime())

        print('%s leaves the server at %.2f.' % (name, env.now))


def setup(env, num_servers, washtime, t_inter):
    """Create a carwash, a number of initial cars and keep creating cars
    approx. every ``t_inter`` minutes."""
    # Create the carwash
    des = DES(env, num_servers, helpingTime)

    # Create 4 initial cars
    for i in range(4):
        env.process(Customer(env, 'Car %d' % i, des))

    # Create more cars while the simulation is running
    while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        i += 1
        env.process(Customer(env, 'Customer %d' % i, des))


# Setup and start the simulation
print('DES')
random.seed(RANDOM_SEED)  # This helps reproducing the results

# Create an environment and start the setup process
env = simpy.Environment()
env.process(setup(env, NUM_servers, helpingTime, T_INTER))

# Execute!
env.run(until=SIM_TIME)