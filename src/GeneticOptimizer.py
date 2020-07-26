import sys
import os
import numpy as np
import random as rnd
from numpy import random as nprnd

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


class GeneticOptimizer:
    def __init__(self, model_class, initI, initR, initN, param_ranges, error_func, real_values, mut_range=0.1,
                 p_mut=0.25, p_mut_ind=0.75, p_regen=0.2, p_cross=0.5, p_cross_ind=0.5, period=15, stop_cond=-1,
                 tournament_size=10,
                 max_gen=2000, start_fitness=-np.inf):
        self.model_class = model_class
        self.initI = initI
        self.initR = initR
        self.initN = initN
        self.param_ranges = param_ranges
        self.error_func = error_func
        self.real_values = real_values
        self.mut_range = mut_range
        self.p_mut = p_mut
        self.p_mut_ind = p_mut_ind
        self.p_regen = p_regen
        self.p_cross = p_cross
        self.p_cross_ind = p_cross_ind
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

        # creator.create("Individual", dict, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.gen_individual, creator.FitnessMax())
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fit_func)
        self.toolbox.register("mate", self.cross_func)
        self.toolbox.register("mutate", self.mut_func)
        # self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        self.toolbox.register("select", tools.selBest)

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
        for param in self.param_ranges:
            model[param] = nprnd.rand() * (self.param_ranges[param][1] - self.param_ranges[param][0]) + \
                           self.param_ranges[param][0]
        model.fitness = fitness
        # return self.model_class(initI=self.initI, initN=self.initN, fitness=fitness, **params)
        return model

    def fit_func(self, model):
        block_print()
        M = self.model_class(initI=self.initI, initR=self.initR, initN=self.initN, **model)
        M.run(T=self.period)
        enable_print()
        predicted_values = list(M.numI[10::10])
        return (-self.error_func(predicted_values=predicted_values, real_values=self.real_values),)

    def mut_func(self, model):
        for param in model:
            if nprnd.rand() < self.p_mut_ind:
                if nprnd.rand() < self.p_regen:
                    model[param] = nprnd.rand() * (self.param_ranges[param][1] - self.param_ranges[param][0]) + \
                                   self.param_ranges[param][0]
                else:
                    model[param] += (self.param_ranges[param][1] - self.param_ranges[param][0]) * \
                                    (nprnd.rand() * self.mut_range * 2 - self.mut_range)
                    model[param] = np.clip(model[param], self.param_ranges[param][0], self.param_ranges[param][1])
        return model

    def cross_func(self, model1, model2):
        for param in self.param_ranges:
            if nprnd.rand() < self.p_cross_ind:
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

    def iteration(self, verbose=True):
        # Begin the evolution
        if self.finished:
            if verbose:
                print("Stop condition is already satisfied!")
            return self.finished, self.best

        # A new generation
        self.g += 1
        if verbose:
            print("-- Generation %i --" % self.g)

        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, self.pop))
        rnd.shuffle(offspring)
        old_pop = list(map(self.toolbox.clone, self.pop))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if nprnd.rand() < self.p_cross:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if nprnd.rand() < self.p_mut:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        self.fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        offspring.extend(old_pop)

        # Select the next generation individuals
        offspring = self.toolbox.select(offspring, len(self.pop))
        self.pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        self.fits = [ind.fitness.values[0] for ind in self.pop]

        length = len(self.pop)
        mean = sum(self.fits) / length
        sum2 = sum(x * x for x in self.fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        self.best = self.pop[np.argmax(self.fits)]
        if verbose:
            print("  Min %s" % min(self.fits))
            print("  Max %s" % max(self.fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("  parameters: {}".format(self.best))

        self.finished = max(self.fits) >= self.stop_cond

        return self.finished, self.best
