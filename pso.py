from models.pso.regions import PSO_Regions


class PSO:
    def __init__(self, img):
        self.img = img
        print(img[0][0])
        self.type = "pso"

    def propose_regions(self):
        regions = PSO_Regions(self.img, popSize=500)
        proposals = regions.after_n_generations(10, 25, 0.05)

        MAX_ASPECT_RATIO = [1.2, 1.4, 1.6, 1.8, 2.0]

        for max_aspect_ratio in reversed(MAX_ASPECT_RATIO):
            candidates = []
            for proposal in proposals:
                if not proposal.aspect_ratio():
                    continue
                if proposal.aspect_ratio() > max_aspect_ratio or (1/proposal.aspect_ratio() > max_aspect_ratio):
                    continue
                if proposal.size() < 60000:
                    continue
                candidates.append(proposal.to_tuple())

        # return(candidates, "pso")
        return(candidates)
