import numpy as np
import cv2
import os

from utils import *
from constants import *

from ga import GA
from pso import PSO


def run_unit_script(approach_class, filename="cat_dog.jpg"):
    img = cv2.imread("{}/{}".format(IMAGE_INPUT_DIR, filename))
    inst = approach_class(img)
    proposals = inst.propose_regions()
    draw_proposal_boxes(img, proposals, save_as="{}/{}/{}".format(IMAGE_OUTPUT_DIR, inst.type, filename))

def run_batch_script(approach_class):
    filenames = [f for f in os.listdir(IMAGE_INPUT_DIR) if f.endswith(".jpg")]
    for filename in filenames:
        run_main_script(approach_class, filename)

run_unit_script(GA)

# Usage:
# 
# run_unit_script(GA, "road.jpg")
# run_batch_script(PSO)