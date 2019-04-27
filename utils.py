import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import datetime
import time
import numpy as np

def draw_proposal_boxes(img, proposals, save_as=None, dpi=500):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for proposal in proposals:
        x, y, w, h = proposal
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    if save_as:
        save_as_dirs = "/".join(save_as.split("/")[:-1])
        if not os.path.isdir(save_as_dirs):
            os.makedirs(save_as_dirs)
        plt.savefig(save_as, bbox_inches='tight')
    else:
        plt.show()

def fetch_timestamp():
    dt = datetime.datetime.today()
    timestamp = int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0)*1000)
    return(timestamp)

def to_mrcnn_rois(proposal_tuples):
    # [(1,2,3,4), (5,6,7,8)]

    count_proposals = len(proposal_tuples)
    # count_tuples = len(proposal_tuples[0])

    rois_in_mrcnn_format = np.zeros((count_proposals, 4), dtype=np.int32)
    for i, (left, top, width, height) in enumerate(proposal_tuples):
        x1 = int(left)
        y1 = int(top)
        x2 = x1 + int(width)
        y2 = y1 + int(height)

        rois_in_mrcnn_format[i][0] = y1
        rois_in_mrcnn_format[i][1] = x1
        rois_in_mrcnn_format[i][2] = y2
        rois_in_mrcnn_format[i][3] = x2

    return(rois_in_mrcnn_format)
