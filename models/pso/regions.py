import pandas as pd
from random import random

from models.base.regions import Regions
from models.pso.region import PSO_Region


class PSO_Regions(Regions):
    def __init__(self, image, popSize=10, maxIterations=10, random_init=True):
        self.image = image
        self.image_height, self.image_width, _ = image.shape
        self.popSize = popSize

        self.gbest = 0.0

        if random_init:
            for i in range(popSize):
                r = PSO_Region(index=i)
                r.randomize(self.image_width, self.image_height)
                r.parent = self
                self.append(r)

    def compute_fitness():
        for region in self:
            self.gbest = max(self.gbest, region.compute_fitness())
        for region in self:
            region.gbest = self.gbest

    def update_regions():
        for region in self:
            region.update_velocity()
            region.update_position()

    def fetch_candidates():
        for i in range(maxIterations):
            for region in self:
                self.compute_fitness()
                self.update_regions()

        return(self)
