import pandas as pd
from random import random

from models.base.regions import Regions
from models.ga.region import GA_Region
import models.misc.selective_search as ss


class GA_Regions(Regions):
    def __init__(self, image, popSize=10, random_init=True):
        self.image = image
        self.image_height, self.image_width, _ = image.shape
        self.popSize = popSize

        if random_init:
            for i in range(popSize):
                r = GA_Region(index=i)
                r.randomize(self.image_width, self.image_height)
                self.append(r)

    def fetch_fitness_values(self):
        return(ss.calc_fitness_of_regions(self.image, self))

    def sort_by_fitness(self):
        fitness_values = self.fetch_fitness_values()
        regions_fitness_tuple = [(self[i], fitness_values[i]) for i in range(len(fitness_values))]

        regions_fitness_tuple = sorted(regions_fitness_tuple, key=lambda t: t[1], reverse=True)
        sorted_regions = GA_Regions(self.image, popSize=self.popSize, random_init=False)
        for (reg, fit) in regions_fitness_tuple:
            sorted_regions.append(reg)
            reg.fitness = fit
        return(sorted_regions)

    def natural_selection(self, eliteSize):
        survivors = GA_Regions(self.image, self.popSize, random_init=False)

        df = pd.DataFrame(self.to_json())
        df['cum_sum'] = df.fitness.cumsum()
        df['cum_perc'] = df.cum_sum/df.fitness.sum()
        
        for i in range(0, eliteSize):
            survivors.append(self[i])
        
        for i in range(0, len(self) - eliteSize):
            pick = random()
            for i in range(0, len(self)):
                # print(df.iat[i,7])
                if pick <= df.iat[i,6]:
                    survivors.append(self[i])
                    break

        print("Natural selection done")
        return(survivors)

    def sbx_crossover(self, parent_1_value, parent_2_value, beta):
        child_1_value = 0.5 * abs(((1 + beta) * parent_1_value) + ((1 - beta) * parent_2_value))
        child_2_value = 0.5 * abs(((1 - beta) * parent_1_value) + ((1 + beta) * parent_2_value))
        return(child_1_value, child_2_value)

    def breed_parents(self, parent_1, parent_2, crossover_operation='sbx', eta=0.9):
        """Executes a simulated binary crossover that modify in-place the input
        individuals. The simulated binary crossover expects :term:`sequence`
        individuals of floating point numbers.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param eta: Crowding degree of the crossover. A high eta will produce
                    children resembling to their parents, while a small eta will
                    produce solutions much more different.
        :returns: A tuple of two individuals.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        # for i, (x1, x2) in enumerate(zip(ind1, ind2)):

        child_1, child_2 = GA_Region(), GA_Region()
        rand = random()
        if rand <= 0.5:
            beta = 2. * rand
        else:
            beta = 1. / (2. * (1. - rand))
        beta **= 1. / (eta + 1.)

        child_1.left, child_2.left = self.sbx_crossover(parent_1.left, parent_2.left, beta)
        child_1.top, child_2.top = self.sbx_crossover(parent_1.top, parent_2.top, beta)
        child_1.width, child_2.width = self.sbx_crossover(parent_1.width, parent_2.width, beta)
        child_1.height, child_2.height = self.sbx_crossover(parent_1.height, parent_2.height, beta)

        child_1.sanitize(self.image_width, self.image_height)
        child_2.sanitize(self.image_width, self.image_height)

        return(child_1, child_2)

    def breed(self):
        i = 0
        parents = self
        children = GA_Regions(self.image, self.popSize, random_init=False)
        
        if (len(parents) % 2) == 1:
            parents = parents[:-1]

        while i < len(parents):
            child_1, child_2 = self.breed_parents(parents[i], parents[i+1])
            child_1.index = i
            child_2.index = i+1
            children.append(child_1)
            children.append(child_2)
            i += 2

        print("Breeding done")

        return(children)

    def mutate(self, p_mut, mu=0, sigma=0.33):
        """This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input individual. This mutation expects a
        :term:`sequence` individual composed of real valued attributes.
        The *indpb* argument is the probability of each attribute to be mutated.
        :param individual: Individual to be mutated.
        :param mu: Mean or :term:`python:sequence` of means for the
                   gaussian addition mutation.
        :param sigma: Standard deviation or :term:`python:sequence` of
                      standard deviations for the gaussian addition mutation.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        This function uses the :func:`~random.random` and :func:`~random.gauss`
        functions from the python base :mod:`random` module.
        """
        generation = GA_Regions(self.image, self.popSize, random_init=False)

        max_perturbation = min(self.image_width, self.image_height)/20
        for region in self:
            generation.append(region.mutate(p_mut, max_perturbation, mu, sigma).sanitize(self.image_width, self.image_height))

        print("Mutation done")
        return(generation)

    def next_generation(self, eliteSize=10, p_mut=0.01):
        parents = self.sort_by_fitness()
        parents = parents.natural_selection(eliteSize)
        offsprings = parents.breed()
        mutants = offsprings.mutate(p_mut)
        return(mutants)

    def after_n_generations(self, n=100, eliteSize=10, p_mut=0.01):
        regions = self
        for g in range(n):
            print("Starting computation for generation {}".format(g))
            regions = regions.next_generation(eliteSize, p_mut)
        return(regions)
