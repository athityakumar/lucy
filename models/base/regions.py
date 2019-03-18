class Regions(list):
    def pretty_print(self):
        fitness = self.fetch_fitness()
        for i in len(self):
            self[i].pretty_print()
            print("Fitness: {}".format(fitness[i]))

    def to_json(self):
        array_of_dicts = []
        for region in self:
            array_of_dicts.append(region.to_json(self.image))
        return(array_of_dicts)

    def print_tuple(self):
        for region in self:
            print(region.to_tuple())

    def size(self):
        return(self.image_width*self.image_height)