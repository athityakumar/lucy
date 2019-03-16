from random import random, gauss

from models.base.region import Region
from models.base.point import Point

import models.fitness.localization as loc
import models.fitness.homogenous_uniformation as hu


class PSO_Region(Region):
    def __init__(self, left=None, top=None, width=None, height=None, index=None):
        super().__init__(left=left, top=top, width=width, height=height, index=index)
        self.pbest = 0.0
        self.gbest = 0.0

    def randomize_left_velocity(self, image_width):
        x_min = min(self.left, image_width - self.left - self.width) / 2.0
        self.v_left = 2 * (random() - 0.5) * x_min

    def randomize_width_velocity(self, image_width):
        x_min = min(self.left, image_width - self.left - self.width) / 2.0
        self.v_width = 2 * (random() - 0.5) * x_min

    def randomize_top_velocity(self, image_height):
        y_min = min(self.top, image_height - self.top - self.height) / 2.0
        self.v_top = 2 * (random() - 0.5) * y_min

    def randomize_height_velocity(self, image_height):
        y_min = min(self.top, image_height - self.top - self.height) / 2.0
        self.v_height = 2 * (random() - 0.5) * y_min

    def randomize(self, image_width, image_height):
        super().randomize(image_width, image_height)

        self.randomize_left_velocity(image_width)
        self.randomize_top_velocity(image_height)
        self.randomize_width_velocity(image_width)
        self.randomize_height_velocity(image_height)

    def update_position():
        self.left += self.v_left
        self.width += self.v_width
        self.top += self.v_top
        self.height += self.v_height

    def compute_velocity_component(self, v, x, w, c1, c2):
        r1, r2 = random(), random()
        return(w*v + c1*r1*(self.pbest-x) + c2*r2*(self.gbest-x))

    def update_velocity(self, w, c1, c2):
        self.v_left = self.compute_velocity_component(self.v_left, self.left, w, c1, c2)
        self.v_top = self.compute_velocity_component(self.v_top, self.top, w, c1, c2)
        self.v_width = self.compute_velocity_component(self.v_width, self.width, w, c1, c2)
        self.v_height = self.compute_velocity_component(self.v_height, self.height, w, c1, c2)

    def compute_fitness():
        fitness_summation = 0

        for x in range(self.left, self.left+self.width):
            for y in range(self.top, self.top+self.height):
                fitness_summation += self.fitness_of_point(x, y)

        self.fitness = fitness_summation/self.size()
        self.pbest = max(self.pbest, self.fitness)
        return(self.fitness)

    def fitness_of_point(x, y):
        return(loc.compute_fitness(self.image, self.parent, x, y))
