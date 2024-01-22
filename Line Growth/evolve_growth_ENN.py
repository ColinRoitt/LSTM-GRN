import random
import os
import datetime
import pickle

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from ELSTM import ELSTM_Dynamic as ELSTM
from ENN import ENN

TARGET = np.array([30, 50])
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = 'output/' + timestamp + '/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def ea(pop_size, gens, cxpb, mutpb, stats, toolbox):
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "avg", "max"

    pop = toolbox.population(n=pop_size)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0]
        ind.moves = fit[1]

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    output_folder = OUTPUT_FOLDER + 'populations/'
    os.makedirs(output_folder, exist_ok=True)

    # Begin the generational process
    for gen in range(1, gens):
        # Save this generation as pickle
        pickle.dump(pop, open(output_folder + f"gen_{gen-1}.pkl", "wb"))
        
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.moves = fit[1]

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, pop_size)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook

def fitness_function(genome, units):
    target = TARGET
    curr_loc = np.array([0., 0.])
    halting = 0
    grn = ENN([units, [5, units]], genome)
    prev_output = None
    moves = [(curr_loc[0], curr_loc[1])]
    for i in range(100):
        loc_and_halt = np.concatenate((curr_loc, [halting]))
        prev_output = grn.forward(loc_and_halt)
        direction = abs(prev_output[0]) * 10
        step_size = abs(prev_output[1]) * 10
        end_devo = prev_output[2] > 0
        if end_devo:
            break
        delta_x = step_size * np.cos(direction)
        delta_y = step_size * np.sin(direction)
        curr_loc[0] += delta_x
        curr_loc[1] += delta_y
        moves.append((curr_loc[0], curr_loc[1]))

    if len(moves) == 1:
        return [1000, 1000], moves
    else:
        return [np.linalg.norm(curr_loc - target), i + 1], moves


def main():
    units = 3
    pop_size = 100
    gens = 100

    cxpb = 0.6
    mutpb = 0.3
    indpb = 0.2
    mu = 0
    sigma = 0.15

    eta = 0.7
    
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti, moves=list)

    print(ENN.get_genome_size([units, [5,8, 5, units]]))

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_float", random.random)
    # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ELSTM.get_genome_size(units))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ENN.get_genome_size([units, [5, 5, units]]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function, units=units)
    # toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", tools.cxSimulatedBinary, eta=eta)
    # Gaussian much better than other options - 0.15 sigma significantly more effective than 0.1
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = ea(pop_size, gens, cxpb, mutpb, stats, toolbox)

    # write experiment parameters to file
    with open(OUTPUT_FOLDER + "params.txt", "w") as f:
        f.write(f"units: {units}\n")
        f.write(f"pop_size: {pop_size}\n")
        f.write(f"gens: {gens}\n")
        f.write(f"cxpb: {cxpb}\n")
        f.write(f"mutpb: {mutpb}\n")
        f.write(f"indpb: {indpb}\n")
        f.write(f"mu: {mu}\n")
        f.write(f"sigma: {sigma}\n")
        f.write(f"eta: {eta}\n")
        f.write(f"TARGET: {TARGET}")

    # save the logbook
    with open(OUTPUT_FOLDER + "logbook.pkl", "wb") as f:
        pickle.dump(log, f)

    # save the population
    with open(OUTPUT_FOLDER + "population.pkl", "wb") as f:
        pickle.dump(pop, f)

    # plot 2d fitness
    fitnesses = [ind.fitness.values for ind in pop]
    fitnesses = np.array(fitnesses)
    plt.scatter(fitnesses[:, 0], fitnesses[:, 1])
    plt.xlabel("Distance from target")
    plt.ylabel("Steps taken")
    plt.savefig(OUTPUT_FOLDER + "pareto.png")
    plt.show()

    # print the above to a file
    with open(OUTPUT_FOLDER + "moves.txt", "w") as f:
        for i, ind in enumerate(pop):
            f.write(f"Individual {i}: {ind.moves}\n")

    print("done")

    return pop, log, hof

if __name__ == "__main__":
    main()