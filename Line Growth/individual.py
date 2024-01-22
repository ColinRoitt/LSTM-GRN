import numpy as np

class individual:
    def __init__(self, genome_size, genome=None):
        self.genome_size = genome_size
        if genome is not None:
            self.genome = genome
        else:
            self.genome = np.random.rand(self.genome_size)
        self.fitness = 0

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def get_genome(self):
        return self.genome

    def set_genome(self, genome):
        self.genome = genome