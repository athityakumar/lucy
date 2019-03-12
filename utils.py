import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
        plt.savefig(save_as, bbox_inches='tight')
    else:
        plt.show()
