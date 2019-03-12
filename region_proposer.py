import numpy as np
import cv2

from utils import *
from constants import *

import ga
import pso


filename = "road.jpg"
img = cv2.imread("{}/{}".format(IMAGE_INPUT_DIR, filename))
proposals, approach_type = ga.propose_regions(img)  # or, pso.propose_regions(img)
draw_proposal_boxes(img, proposals, save_as="{}/{}/{}".format(IMAGE_OUTPUT_DIR, approach_type, filename))
