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
# from ELSTM_v2 import ELSTM

from grow_tree import fitness_function, draw_and_save

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = 'output_2/' + timestamp + '/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# from shapes.letter_z import dot_locations
# targets = np.array(dot_locations)
# targets = np.array([(299, 299), (500, 255), (500, 785)])
fixed = False

def fitness_function_random_points(genome, units):
    fitnesses = [fitness_function(genome, units) for _ in range(5)]
    fitnesses = np.array(fitnesses)
    return np.mean(fitnesses, axis=0)

def fitness_function_fixed_points(genome, units):
    return fitness_function(genome, units, targets)

def ea(pop_size, gens, cxpb, mutpb, stats, toolbox):
    # Initialize statistics object          
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "avg", "max"

    pop = toolbox.population(n=pop_size)
    # pop = pickle.load(open("output/2023-10-05_10-25-36/populations/gen_98.pkl", "rb"))

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

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
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, pop_size)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook

def main():
    units = 6
    pop_size = 50
    gens = 100

    cxpb = 0.5
    mutpb = 0.3
    indpb = 0.2
    mu = 0
    sigma = 0.30

    eta = 0.4
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti, moves=list)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_float", random.random)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ELSTM.get_genome_size(units, outputs=units))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    if fixed:
        toolbox.register("evaluate", fitness_function_fixed_points, units=units)
    else:
        toolbox.register("evaluate", fitness_function_random_points, units=units)
    # toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", tools.cxSimulatedBinary, eta=eta)
    # Gaussian much better than other options - 0.15 sigma significantly more effective than 0.1
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)

    # pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = ea(pop_size, gens, cxpb, mutpb, stats, toolbox)
    # pop = pickle.load(open("output/2023-10-12_14-36-33/populations/gen_98.pkl", "rb"))

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

    # save the logbook
    # with open(OUTPUT_FOLDER + "logbook.pkl", "wb") as f:
    #     pickle.dump(log, f) 

    # save the population
    with open(OUTPUT_FOLDER + "population.pkl", "wb") as f:
        pickle.dump(pop, f)

    # plot 2d fitness
    fitnesses = [ind.fitness.values for ind in pop]
    fitnesses = np.array(fitnesses)
    plt.scatter(fitnesses[:, 0], fitnesses[:, 1])
    plt.xlabel("Targets hit")
    plt.ylabel("Steps taken")
    plt.savefig(OUTPUT_FOLDER + "pareto.png")
    # plt.show(block=False)
    # plt.pause(0.001)

    # 3d scatter
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(fitnesses[:, 0], fitnesses[:, 1], fitnesses[:, 2])
    # ax.set_xlabel("Targets hit")
    # ax.set_ylabel("Steps taken")
    # ax.set_zlabel("Non-capturing movements")
    # plt.savefig(OUTPUT_FOLDER + "pareto_3d.png")
    # plt.show(block=False)
    # plt.pause(0.001)

    os.makedirs(OUTPUT_FOLDER + "final_individuals/", exist_ok=True)
    os.makedirs(OUTPUT_FOLDER + "devo/", exist_ok=True)
    # draw each individual if it moves more than one space
    # if fixed:
    #     to_test = 1
    # else:
    #     to_test = 3
    to_test = 1
    with open(OUTPUT_FOLDER + "moves.txt", "a") as f:
        for n in range(to_test):
            for i, ind in enumerate(pop):
                if ind.fitness.values[1] > 1 or ind.fitness.values[0] > 0:
                    print(f"Drawing individual {i} {ind.fitness.values}")
                    file_path = OUTPUT_FOLDER + f"final_individuals/individual_{i}_{n}.png"
                    t = targets if fixed else None
                    _, _, moves = draw_and_save(ind, units, file_path, targets=t, output_folder = OUTPUT_FOLDER, index = i)
                    # append moves to move file
                    formatted_moves = [f"{m[0][0]} {m[1][0]} [{m[2][0]} {m[2][1]}]" for m in moves]
                    f.write(f"Individual {i}_{n}:\n")
                    f.write("\n".join(formatted_moves))
                    f.write("\n")

    print("done")
    # plt.show()

    return pop, log, hof

if __name__ == "__main__":
    main()