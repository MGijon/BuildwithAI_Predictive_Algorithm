import numpy as np
from numpy import random as rnd

from deap import base
from deap import creator
from deap import tools

import sys, os

from seirsplus.models import *


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


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


class SEIRSModelGen(SEIRSModel):
    def __init__(self, initI, initN, fitness=-np.inf, **params):
        super().__init__(initI=initI, initN=initN, **params)
        self.fitness = fitness


class GeneticOptimizer:
    def __init__(self, model_class, initI, initN, params, param_ranges, error_func, real_values, mut_range=0.05,
                 p_mut=0.2, p_ind=0.5, p_regen=0.1, p_cross=0.5, period=15, stop_cond=-1, tournament_size=10,
                 max_gen=2000, start_fitness=-np.inf):
        self.model_class = model_class
        self.initI = initI
        self.initN = initN
        self.params = params
        self.param_ranges = param_ranges
        self.error_func = error_func
        self.real_values = real_values
        self.mut_range = mut_range
        self.p_mut = p_mut
        self.p_ind = p_ind
        self.p_regen = p_regen
        self.p_cross = p_cross
        self.period = period
        self.stop_cond = stop_cond
        self.max_gen = max_gen
        self.start_fitness = start_fitness

        self.pop = list()
        self.fits = list()
        self.fitnesses = list()
        self.g = 0
        self.best = None
        self.finished = False

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", dict, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.gen_individual, creator.FitnessMax())
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fit_func)
        self.toolbox.register("mate", self.cross_func)
        self.toolbox.register("mutate", self.mut_func)
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    class Individual(dict):
        def __init__(self):
            super().__init__()
            self.fitness = -np.inf


    def gen_individual(self, fitness):
        # params = dict()
        # for param in self.params:
        #     params[param] = rnd.rand() * (self.param_ranges[param][1] - self.param_ranges[param][0]) +\
        #                     self.param_ranges[param][0]
        model = self.Individual()
        for param in self.params:
            model[param] = rnd.rand() * (self.param_ranges[param][1] - self.param_ranges[param][0]) +\
                            self.param_ranges[param][0]
        model.fitness = fitness
        # return self.model_class(initI=self.initI, initN=self.initN, fitness=fitness, **params)
        return model

    def fit_func(self, model):
        blockPrint()
        M = self.model_class(initI=self.initI, initN=self.initN, **model)
        M.run(T=self.period)
        enablePrint()
        predicted_values = list(M.total_num_infections()[10::10])
        return (-self.error_func(predicted_values=predicted_values, real_values=self.real_values),)

    def mut_func(self, model):
        for param in model:
            if rnd.rand() < self.p_ind:
                if rnd.rand() < self.p_regen:
                    model[param] = rnd.rand() * (self.param_ranges[param][1] - self.param_ranges[param][0]) + \
                                   self.param_ranges[param][0]
                else:
                    model[param] += (self.param_ranges[param][1] - self.param_ranges[param][0]) *\
                                    (rnd.rand() * self.mut_range * 2 - self.mut_range)
                    model[param] = np.clip(model[param], self.param_ranges[param][0], self.param_ranges[param][1])

    def cross_func(self, model1, model2):
        for param in self.params:
            if rnd.rand() < self.p_ind:
                model1[param], model2[param] = model2[param], model1[param]
        return model1, model2

    def initialize(self, population=100):
        self.pop = self.toolbox.population(n=population)
        self.fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit

        # Extracting all the fitnesses of
        self.fits = [ind.fitness.values[0] for ind in self.pop]

        # Variable keeping track of the number of generations
        self.g = 0

    def iteration(self):
        # Begin the evolution
        if self.finished:
            print("Stop condition is already satisfied!")
            return self.finished, self.best

        # A new generation
        self.g += 1
        print("-- Generation %i --" % self.g)

        # Select the next generation individuals
        offspring = self.toolbox.select(self.pop, len(self.pop))

        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rnd.rand() < self.p_cross:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if rnd.rand() < self.p_mut:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        self.fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        self.pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        self.fits = [ind.fitness.values[0] for ind in self.pop]

        length = len(self.pop)
        mean = sum(self.fits) / length
        sum2 = sum(x * x for x in self.fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(self.fits))
        print("  Max %s" % max(self.fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        self.best = self.pop[np.argmax(self.fits)]
        print("  parameters: {}".format(self.best))

        self.finished = max(self.fits) >= self.stop_cond

        return self.finished, self.best


    # print("Best params: ({:.3f}, {:.3f})\nBest prediction: {}\nTrue values: {}".format(best.beta, best.gamma, best.data, DATA))


# def generate_data(beta=0.1, gamma=0.01, S=9900, I=100, R=0, T=14):
#     N = S + I + R
#     data = list()
#     data.append((S, I, R))
#     for t in range(T):
#         rbeta = beta * (1. + rnd.rand() * DATA_NOISE * 2 - DATA_NOISE)
#         rgamma = gamma * (1. + rnd.rand() * DATA_NOISE * 2 - DATA_NOISE)
#         Snew = S - rbeta * I * S / N
#         Inew = I + rbeta * I * S / N - rgamma * I
#         Rnew = R + rgamma * I
#
#         S = Snew
#         I = Inew
#         R = Rnew
#
#         data.append((S, I, R))
#
#     return data
