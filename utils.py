import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import datetime
import time


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
