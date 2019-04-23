import pandas as pd
from random import random
from copy import copy, deepcopy

from models.base.regions import Regions
from models.pso.region import PSO_Region

import models.fitness.localization as loc
import models.fitness.homogenous_uniformation as hu


class PSO_Regions(Regions):
    def __init__(self, image, popSize=10, maxIterations=10, random_init=True):
        self.image = image
        self.image_height, self.image_width, _ = image.shape
        self.popSize = popSize
        self.maxIterations = maxIterations
        self.gbest_score = 0.0

        if random_init:
            for i in range(popSize):
                r = PSO_Region(index=i)
                r.parent = self
                r.randomize(self.image_width, self.image_height)
                self.append(r)

        self.gbest = self[0]
        self.pixel_wise_fitness = self.calc_pixel_wise_fitness()
        self.__repr__(debug=True)

    def calc_pixel_wise_fitness(self):
        fit = []
        for x in range(0, self.image_width):
            fit_row = []
            for y in range(0, self.image_height):
                fit_row.append(loc.compute_fitness(self.image, self, x, y))
            fit.append(fit_row)
        return(fit)

    def compute_fitness(self):
        for region in self:
            fit = region.compute_fitness()
            if fit > self.gbest_score:
                self.gbest_score = fit
                self.gbest = region

    def update_regions(self, w, c1, c2):
        for region in self:
            region.update_velocity(w, c1, c2)
        self.__repr__(debug=True, heading="After updating velocity")

        for region in self:
            region.update_position()
        self.__repr__(debug=True, heading="After updating position")

            # print("Before sanitize:")
            # region.__repr__()
        for region in self:
            region.sanitize(self.image_width, self.image_height)
        self.__repr__(debug=True, heading="After sanitize")

            # print("After sanitize:")
            # region.__repr__()

    def fetch_candidates(self, w=1.0, c1=0.5, c2=0.5):
        for i in range(self.maxIterations):
            print("Generation {} computation started".format(i))
            dynamic_w = w - 0.5 * i / self.maxIterations
            # print(dynamic_w, c1, c2)

            self.compute_fitness()
            self.__repr__(debug=True, heading="Before updating")
            self.update_regions(dynamic_w, c1, c2)

            print("Generation {} computation ended".format(i))
        return(self)

    def __repr__(self, debug=False, heading=""):
        print(heading)

        if debug:
            for region in self[0:5]:
                region.__repr__()
        else:
            for region in self:
                region.__repr__()
