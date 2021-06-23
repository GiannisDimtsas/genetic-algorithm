import pandas as pd
import numpy as np
from deap import base, creator,tools
import random
from matplotlib import pyplot
import tensorflow as tf
from network import create_model,train

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Genetic Algorithm parameters
pop_size=10
generations=20
crossover_prop=0.6
mutation_prop=0.0
individuals = []

#set toolbox
def set_toolbox():
    global toolbox
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    toolbox = base.Toolbox()
    toolbox.register("select", tools.selRoulette,fit_attr='fitness')
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.5)
    toolbox.register("evaluate", cross_entropy)


#define initial population
def initial_pop():
    for i in range(pop_size):
        individuals.append(create_model())
    return individuals

#main
def main():
    set_toolbox()

    #Create population
    individuals = initial_pop()
    individuals,losses = train(individuals)

    for i in range(generations):
        offspring = toolbox.select(individuals, len(individuals))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prop:
                print('Crossover is done')
                toolbox.mate(child1, child2)

        for mutant in offspring:
            if random.random() < mutation_prop:
                print('mutation is done')
                toolbox.mutate(mutant)


        individuals[:] = offspring

        best = individuals[np.argmax([toolbox.evaluate(x) for x in individuals])]

        print(best)

main()