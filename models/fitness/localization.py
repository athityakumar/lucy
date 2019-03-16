def compute_fitness(img, particles, x, y):
    return((punctual_fitness(img, x, y) + local_fitness(particles, x, y)) / 2.0)

def local_fitness(particles, x, y):
    neighbours = 0
    for region in particles:
        if region.contains_point(x, y):
            neighbours += 1
    return(neighbours / len(particles))

def punctual_fitness(img, x, y):
    r, g, b = img[x][y]
    if ((abs(r-g) < 30) and (abs(r-b) > 60)):
        return(30 - abs(r-g))
    else:
        return(0.0)
