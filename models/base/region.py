from random import random


class Region:
    def __init__(self, left=None, top=None, width=None, height=None, index=None):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.fitness = 0.0
        self.index = index

    def size(self):
        return(self.width*self.height)

    def aspect_ratio(self):
        if self.height == 0 or self.width == 0:
            return(None)
        return(self.width/self.height)

    def randomize_left(self, image_width):
        self.left = random() * image_width

    def randomize_width(self, image_width):
        self.width = random() * (image_width - self.left)

    def randomize_top(self, image_height):
        self.top = random() * image_height

    def randomize_height(self, image_height):
        self.height = random() * (image_height - self.top)

    def randomize(self, image_width, image_height):
        self.randomize_left(image_width)
        self.randomize_top(image_height)
        self.randomize_width(image_width)
        self.randomize_height(image_height)

    def pretty_print(self):
        print("Left: {}, Top: {}, Width: {}, Height: {}".format(self.left, self.top, self.width, self.height))

    def to_json(self, image):
        return({
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height,
            'fitness': self.fitness
            })

    def to_tuple(self):
        return((self.left, self.top, self.width, self.height))

    def sanitize(self, image_width, image_height):
        if self.left < 0:
            self.width += self.left
            self.left = 0.0
        if self.left > image_width:
            self.left = image_width
            self.width = 0.0
        if self.width < 0:
            self.width = 0.0
        if self.left + self.width > image_width:
            self.width = image_width - self.left

        if self.top < 0:
            self.height += self.top
            self.top = 0.0
        if self.top > image_height:
            self.top = image_height
            self.height = 0.0
        if self.height < 0:
            self.height = 0.0
        if self.top + self.height > image_height:
            self.height = image_height - self.top
        return(self)

    def contains_point(self, x, y):
        contains_x = x < self.left and x > self.left+self.width
        contains_y = y < self.top and y > self.top+self.height
        return(contains_x and contains_y)