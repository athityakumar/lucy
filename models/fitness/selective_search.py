# -*- coding: utf-8 -*-
from __future__ import division

# # Refer https://github.com/ezstoltz/genetic-algorithm/blob/master/genetic_algorithm_TSP.ipynb
import numpy, random, operator, pandas as pd, matplotlib.pyplot as plt


import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#
#  - Modified version with LBP extractor for texture vectorization


def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """

    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

        extract HSV
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image

        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.

        output will be [height(*)][width(*)]
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region

        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # mask by the colour channel
        fd = img[:, colour_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)

    return hist


def _extract_regions(img, regions):

    R = {}

    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    img_height, img_width, img_channel = img.shape

    for i,region in enumerate(regions):
        R[i] = {
            "min_x": region.left, "min_y": region.top,
            "max_x": region.left+region.width, "max_y": region.top+region.height, "labels": [i]}

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R

def _extract_region(img, region):

    R = {}

    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    img_height, img_width, img_channel = img.shape

    for region in regions:
        R[i] = {
            "min_x": region.left, "min_y": region.top,
            "max_x": region.left+region.width, "max_y": region.top+region.height, "labels": [i]}

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R

def _extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt

def calc_fitness_of_regions(im_orig, regions):
    assert im_orig.shape[2] == 3, "3ch image is expected"

    scale=1.0
    sigma=0.8
    min_size=50

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]

    # imgR = {
    #         "min_x": 0xffff, "min_y": 0xffff,
    #         "max_x": img.shape[1], "max_y": img.shape[0], "labels": [-1]}

    # # pass 2: calculate texture gradient
    # tex_grad = _calc_texture_gradient(img)

    # # pass 3: calculate colour histogram of each region
    # hsv = skimage.color.rgb2hsv(im_orig[:, :, :3])

    # # colour histogram
    # masked_pixels = hsv[:, :, :][img[:, :, 3] == -1]
    # imgR["size"] = len(masked_pixels / 4)
    # imgR["hist_c"] = _calc_colour_hist(masked_pixels)

    # # texture histogram
    # imgR["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == -1])

    R = _extract_regions(img, regions)
    # fitness_of_regions = [_calc_sim(r, imgR, imsize) for r in R]

    fitness_of_regions = []
    for i in range(len(R)):
        fitness = 0.0
        for j in range(len(R)):
            if i != j:
                fitness += _calc_sim(R[i], R[j], imsize)
        fitness_of_regions.append(fitness)

    return(fitness_of_regions)

# def calc_fitness_of_region(im_orig, region):
#     assert im_orig.shape[2] == 3, "3ch image is expected"

#     # load image and get smallest regions
#     # region label is stored in the 4th value of each pixel [r,g,b,(region)]
#     img = _generate_segments(im_orig, scale, sigma, min_size)

#     if img is None:
#         return None, {}

#     imsize = img.shape[0] * img.shape[1]

#     imgR = {
#             "min_x": 0xffff, "min_y": 0xffff,
#             "max_x": img.shape[1], "max_y": img.shape[0], "labels": [-1]}

#     # pass 2: calculate texture gradient
#     tex_grad = _calc_texture_gradient(img)

#     # pass 3: calculate colour histogram of each region

#     # colour histogram
#     masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
#     imgR["size"] = len(masked_pixels / 4)
#     imgR["hist_c"] = _calc_colour_hist(masked_pixels)

#     # texture histogram
#     imgR["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == -1])

#     fitness_of_region = _calc_sim(region, imgR, imsize)

#     return(fitness_of_region)
