from random import random, gauss

from models.base.region import Region


class GA_Region(Region):
    def mutate_individual_value(self, individual, p_mut, max_perturbation, mu, sigma):
        if random() < p_mut:
            deviation = gauss(mu, sigma)
            if (individual+deviation) > 0:
                individual += deviation

        return(individual)

    def mutate(self, p_mut, max_perturbation, mu=0, sigma=0.33):
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
        left = self.mutate_individual_value(self.left, p_mut, max_perturbation, mu, sigma)
        top = self.mutate_individual_value(self.top, p_mut, max_perturbation, mu, sigma)
        width = self.mutate_individual_value(self.width, p_mut, max_perturbation, mu, sigma)
        height = self.mutate_individual_value(self.height, p_mut, max_perturbation, mu, sigma)
        return(GA_Region(left=left, top=top, width=width, height=height))
