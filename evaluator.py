from region_proposer import *

# TODO:
# 
# Import COCO dataset and 
# R-CNN code:
# https://github.com/rbgirshick/rcnn
# https://github.com/broadinstitute/keras-rcnn
# https://github.com/PatrickXYS/Reproduce_frcnn
# https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN
# https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/train.py
# https://github.com/yangxue0827/RCNN
# https://github.com/s-gupta/rcnn-depth
# https://github.com/taolei87/rcnn/tree/master/code
# https://github.com/ijkguo/mx-rcnn
#  
# Replace selective search call
# by GA/PSO functions to get
# Regions of Interest (RoIs)
# 
# Run the evaluation script for
# COCO dataset, from the weights
# or results obtained
#
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
# 
# Compare the mAP values with the
# standard baseline ones
# 
# (Might have to run the baseline
# too once at the local GPU system,
# to make sure that comparisions are
# meaningful and not biased due to
# different hardware)
# 
# Also, if possible, record the frames
# per second metric of both baseline and
# GA/PSO.
# 
# The end.

def train_dataset(approach_class):
    training_image_paths = []
    for training_image_path in training_images:
        training_image = get_img_from_path(training_image_path)
        inst = approach_class(training_image)
        proposals = inst.propose_regions()

def test_dataset():

