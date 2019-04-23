from models.ga.regions import GA_Regions


class GA:
    """
    Wrapper around the base GA class, that proposes regions
    based on the fitness value mentioned in selective search,
    and does aspect-ratio-based filtering of regions as well.
    """

    def __init__(self, img):
        self.img = img
        self.type = "ga"
        self.img_height, self.img_width, _ = img.shape
        self.img_size = self.img_width * self.img_height

    def propose_regions(self):
        regions = GA_Regions(self.img, popSize=500)
        proposals = regions.after_n_generations(10, 25, 0.05)

        MIN_ASPECT_RATIO = 2.0

        candidates = []
        for proposal in proposals:
            if not proposal.aspect_ratio():
                continue
            if proposal.aspect_ratio() > MIN_ASPECT_RATIO or (1/proposal.aspect_ratio() > MIN_ASPECT_RATIO):
                continue
            if proposal.size() < 0.05 * self.img_size:
                continue
            if proposal.left == 0.0 or proposal.top == 0.0 or proposal.bottom() == self.img_height or proposal.right() == self.img_width:
                continue
            candidates.append(proposal.to_tuple())

        return(candidates)
