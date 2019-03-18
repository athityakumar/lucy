from models.pso.regions import PSO_Regions


class PSO:
    def __init__(self, img):
        self.img = img
        self.type = "pso"

    def propose_regions(self):
        regions = PSO_Regions(self.img, popSize=200, maxIterations=1)
        proposals = regions.fetch_candidates(w=1.0, c1=1, c2=1)

        MAX_ASPECT_RATIO = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

        # candidates = [proposal.to_tuple() for proposal in proposals]
        # candidates = []
        # for proposal in proposals:
        #     if proposal.size() != 0:
        #         candidates.append(proposal.to_tuple())

        for max_aspect_ratio in reversed(MAX_ASPECT_RATIO):
            candidates = []
            for proposal in proposals:
                if not proposal.aspect_ratio():
                    continue

                print(proposal.aspect_ratio(), 1.0/proposal.aspect_ratio(), proposal.size())

                if proposal.aspect_ratio() > max_aspect_ratio or (1/proposal.aspect_ratio() > max_aspect_ratio):
                    continue
                if proposal.size() < 60000:
                    continue

                # proposal.__repr__()
                candidates.append(proposal.to_tuple())

        return(candidates)
