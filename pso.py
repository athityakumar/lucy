from models.pso.regions import PSO_Regions


class PSO:
    """
    Wrapper around the base PSO class, that proposes regions
    based on the fitness value mentioned in selective search,
    and does aspect-ratio-based filtering of regions as well.
    """

    def __init__(self, img):
        self.img = img
        self.type = "pso"
        self.img_height, self.img_width, _ = img.shape
        self.img_size = self.img_width * self.img_height

    def propose_regions(self, force=False):
        regions = PSO_Regions(self.img, popSize=300, maxIterations=0)
        proposals = regions.fetch_candidates(w=1.0, c1=2.0, c2=2.0)

        MIN_ASPECT_RATIO = 2.0

        candidates = []
        for proposal in proposals:
            if not force:
                if not proposal.aspect_ratio():
                    continue
                if proposal.aspect_ratio() > MIN_ASPECT_RATIO or (1/proposal.aspect_ratio() > MIN_ASPECT_RATIO):
                    continue
                if proposal.size() < 0.05 * self.img_size:
                    continue
                if proposal.left == 0.0 or proposal.top == 0.0 or proposal.bottom() == self.img_height or proposal.right() == self.img_width:
                    continue
                candidates.append(proposal.to_tuple())
            else:
                candidates.append(proposal.to_tuple())
        return(candidates)
