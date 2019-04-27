from random import random, gauss
from copy import copy, deepcopy

from models.base.region import Region
from models.base.point import Point

import models.fitness.localization as loc
import models.fitness.homogenous_uniformation as hu


class PSO_Region(Region):
    def __init__(self, left=None, top=None, width=None, height=None, index=None):
        super().__init__(left=left, top=top, width=width, height=height, index=index)
        self.pbest = self
        self.pbest_score = 0.0

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

    def update_position(self):
        self.left += self.v_left
        self.width += self.v_width
        self.top += self.v_top
        self.height += self.v_height

    def compute_velocity_component(self, v, x, pbest, gbest, w, c1, c2):
        r1, r2 = random(), random()
        return(w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x))

    def update_velocity(self, w, c1, c2):
        v_left_new = self.compute_velocity_component(self.v_left, self.left, self.pbest.left, self.parent.gbest.left, w, c1, c2)
        v_top_new = self.compute_velocity_component(self.v_top, self.top, self.pbest.top, self.parent.gbest.top, w, c1, c2)
        v_width_new = self.compute_velocity_component(self.v_width, self.width, self.pbest.width, self.parent.gbest.width, w, c1, c2)
        v_height_new = self.compute_velocity_component(self.v_height, self.height, self.pbest.height, self.parent.gbest.height, w, c1, c2)

        self.v_left = v_left_new
        self.v_top = v_top_new
        self.v_width = v_width_new
        self.v_height = v_height_new
        # self.v_left = self.correct_if_does_not_lie_in_range(-1.0*self.left, 0.1*self.parent.image_width, self.parent.image_width-self.left, self.v_left-v_left_new)
        # self.v_top = self.correct_if_does_not_lie_in_range(-1.0*self.top, 0.1*self.parent.image_height, self.parent.image_height-self.top, self.v_top-v_top_new)
        # self.v_width = self.correct_if_does_not_lie_in_range(-1.0*self.width, 0.1*self.parent.image_width, self.parent.image_width-self.left-self.width, self.v_width-v_width_new)
        # self.v_height = self.correct_if_does_not_lie_in_range(-1.0*self.height, 0.1*self.parent.image_height, self.parent.image_height-self.top-self.height, self.v_height-v_height_new)

    def correct_if_does_not_lie_in_range(self, minimum, max1, max2, value):
        maximum = min(max1, max2)
        minimum = min(minimum, -1.0*maximum)
        value = abs(value)
        if value < minimum:
            value = minimum
        if value > maximum:
            value = maximum
        return(value)

    def compute_fitness(self):
        fitness_summation = 0

        for x in range(int(self.left), int(self.left+self.width)):
            for y in range(int(self.top), int(self.top+self.height)):
                # fitness_summation += self.fitness_of_point(x, y)
                fitness_summation += self.parent.pixel_wise_fitness[x][y]

        # size = self.size() or self.width or self.height or 1
        # size = 1.0
        self.fitness = fitness_summation/self.perimeter()
        if self.fitness > self.pbest_score:
            self.pbest_score = self.fitness
            self.pbest = self

        return(self.fitness)

    def fitness_of_point(self, x, y):
        return(loc.compute_fitness(self.parent.image, self.parent, x, y))

    def sanitize(self, image_width, image_height):
        super().sanitize(image_width, image_height)

        if self.left == image_width:
            self.randomize_left(image_width)
        if self.top == image_height:
            self.randomize_top(image_height)
        if self.width == 0.0:
            self.randomize_width(image_width)
        if self.height == 0.0:
            self.randomize_height(image_height)

        # if self.left == image_width or self.top == image_height or self.width == image_width or self.height == image_height:
        #     self.randomize(image_width, image_height)

        # if self.left + self.v_left < 0:
        #     self.v_left =  -1 * min(self.left, self.v_left)

        # if self.left + self.width >= image_width:
        #     self.v_left = -1 * abs(self.v_left)
        #     self.v_width = abs(self.v_width)

        # if self.width + self.v_width > image_width:
        #     self.v_width *= -1

        # if self.top + self.v_top < 0:
        #     self.v_top =  -1 * min(self.top, self.v_top)
        # if self.top + self.height >= image_height:
        #     self.v_top = -1 * abs(self.v_top)
        #     self.v_height = abs(self.v_height)
        # if self.height + self.v_height > image_height:
        #     self.v_height *= -1

        # if self.width == 0:
        #     self.width = random() * (image_width - self.left)
        # if self.height == 0:
        #     self.height = random() * (image_height - self.top)

        return(self)


    def __repr__(self):
        print("Left: {}, Top: {}, Width: {}, Height: {}, V_left: {}, V_top: {}, V_width: {}, V_height: {}, Pbest: {}, Gbest: {}".format(self.left, self.top, self.width, self.height, self.v_left, self.v_top, self.v_width, self.v_height, self.pbest_score, self.parent.gbest_score))