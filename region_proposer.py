# import numpy as np
import cv2
import os

from utils import *
from constants import *
import cli

from ga import GA
from pso import PSO


def run_unit_script(approach_class, filename="cat_dog.jpg", timestamp=None):
    if not os.path.exists(filename):
        filepath = "{}/{}".format(IMAGE_INPUT_DIR, filename)
    else:
        filepath = filename
        filename = filepath.split("/")[-1]

    img = cv2.imread(filepath)
    inst = approach_class(img)
    proposals = inst.propose_regions()
    if not timestamp:
        timestamp = fetch_timestamp()

    draw_proposal_boxes(img, proposals, save_as="{}/{}/{}/{}".format(IMAGE_OUTPUT_DIR, inst.type, timestamp, filename))

def run_batch_script(approach_class, dirname=IMAGE_INPUT_DIR):
    filenames = [f for f in os.listdir(dirname) if f.endswith(".jpg")]
    timestamp = fetch_timestamp()
    for filename in filenames:
        run_unit_script(approach_class, filename=filename, timestamp=timestamp)

args = cli.fetch_args()
print(args)

pipeline = GA if args.pipeline == "ga" else PSO

# TODO:
# 
# Checks for `if not args.log` 
# If log file already exists,
# delete it and then append all
# print statement outputs to this
# log filepath.
# 
# Like this:
# https://github.com/Demfier/PsyNLP/blob/master/psynlp/helpers/builtins.py#L4:5
# 
# Or, just directly use the logging
# module.

if args.mode == "unit":
    run_unit_script(pipeline, filename=args.filename)
else:
    run_batch_script(pipeline, dirname=args.filename)
