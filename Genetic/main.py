import numpy as np
from numpy import random as rnd

from deap import base
from deap import creator
from deap import tools


rnd.seed(1138)


DATA_NOISE = 0
BETA_RANGE = (0.001, 0.2)
GAMMA_RANGE = (0.001, 0.5)
MUT_RANGE = 0.05
P_MUT = 0.2
P_REGEN = 0.1
P_CROSS = 0.5
# S0 = 9900
# I0 = 100
# R0 = 0
PERIOD = 14
STOP_COND = -0.1
MAX_GEN = 2000


class SIR:
    def __init__(self, S0, I0=0, R0=0, beta0=1., gamma0=1., T=14, fitness=None):
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.S = S0
        self.I = I0
        self.R = R0
        self.N = S0 + I0 + R0
        self.t = 0
        self.beta = beta0
        self.gamma = gamma0
        self.data = [(S0, I0, R0)]
        if T > 0:
            for t in range(T):
                self.next()
        self.fitness = fitness

    def copy(self):
        sir = SIR(self.S, self.I, self.R, self.beta, self.gamma)
        sir.t = self.t
        sir.data = self.data.copy()

    def next(self):
        S = self.S - self.beta * self.I * self.S / self.N
        I = self.I + self.beta * self.I * self.S / self.N - self.gamma * self.I
        R = self.R + self.gamma * self.I

        self.S = S
        self.I = I
        self.R = R

        self.t += 1

        self.data.append((self.S, self.I, self.R))

        return self.S, self.I, self.R

    def recalc(self, T):
        self.S = self.S0
        self.I = self.I0
        self.R = self.R0

        self.data = [(self.S0, self.I0, self.R0)]
        for t in range(T):
            self.next()


def generate_data(beta=0.1, gamma=0.01, S=9900, I=100, R=0, T=14):
    N = S + I + R
    data = list()
    data.append((S, I, R))
    for t in range(T):
        rbeta = beta * (1. + rnd.rand() * DATA_NOISE * 2 - DATA_NOISE)
        rgamma = gamma * (1. + rnd.rand() * DATA_NOISE * 2 - DATA_NOISE)
        Snew = S - rbeta * I * S / N
        Inew = I + rbeta * I * S / N - rgamma * I
        Rnew = R + rgamma * I

        S = Snew
        I = Inew
        R = Rnew

        data.append((S, I, R))

    return data


DATA = generate_data()


def genInd(S0, I0, R0, beta_range, gamma_range, T, fitness):
    return SIR(S0, I0, R0,
               rnd.rand() * (beta_range[1] - beta_range[0]) + beta_range[0],
               rnd.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0],
               T,
               fitness)


def fitFunc(sir, data):
    # print(sir.data)
    N = len(data)
    sir.recalc(N)

    if len(sir.data) < N:
        for i in range(N - len(sir.data)):
            sir.next()

    error = 0
    for i in range(N):
        for j in range(3):
            error += ((data[i][j] - sir.data[i][j])**2) / (3 * N)

    return (-error,)


def mutFunc(sir, p_regen, beta_range, gamma_range, mut_range):
    if rnd.rand() < p_regen:
        sir.beta = rnd.rand() * (beta_range[1] - beta_range[0]) + beta_range[0]
        sir.gamma = rnd.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    else:
        sir.beta += rnd.rand() * mut_range * 2 - mut_range
        sir.gamma += rnd.rand() * mut_range * 2 - mut_range

def crossFunc1(sir1, sir2):
    sir1.beta, sir2.beta = sir2.beta, sir1.beta
    return sir1, sir2

# def crossFunc2(sir1, sir2):
#


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", SIR, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", genInd, 9900, 100, 0, BETA_RANGE, GAMMA_RANGE, PERIOD, creator.FitnessMax())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitFunc, data=DATA)
toolbox.register("mate", crossFunc1)
toolbox.register("mutate", mutFunc, p_regen=P_REGEN, beta_range=BETA_RANGE, gamma_range=GAMMA_RANGE, mut_range=MUT_RANGE)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=100)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations

print("\n\n\n\n\n=======================================")
g = 0
print(DATA)
print(pop[0].data)
sir = SIR(9900, 100, 0, 0.1, 0.01)
print(sir.data)
print(fitFunc(sir,DATA))


# Begin the evolution
while max(fits) < STOP_COND and g < MAX_GEN:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if rnd.rand() < P_CROSS:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if rnd.rand() < P_MUT:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    best = pop[np.argmax(fits)]
    print("  beta: {:.3f}, gamma: {:.3f}".format(best.beta, best.gamma))

print("Best params: ({:.3f}, {:.3f})\nBest prediction: {}\nTrue values: {}".format(best.beta, best.gamma, best.data, DATA))
