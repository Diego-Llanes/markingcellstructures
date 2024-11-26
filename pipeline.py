import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List, Tuple, Dict
from collections import namedtuple

from functions import find_best_zslices, threshold_image, find_clusters

DATA_DIR = Path("/research/jagodzinski/markingcellstructures")

"""
notes:
 - data is shaped as (z, c, y, x)
"""


def main():
    files = list(DATA_DIR.rglob("*.tif"))
    sample = files[0]
    img = tifffile.imread(sample)
    
    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]
    
    custom_colors = ["lightgray", "cyan", "blue", "red", "green", "yellow", "orange", "purple"]
    cmap = ListedColormap(custom_colors)
    boundaries = [-2, -1, 0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(boundaries, cmap.N)

    # find best z_slice (might just be 24 ??)
    zslices= []
    for channel_idx, channel in enumerate(channels):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

        channel_img = img[:, channel_idx]
        zslice = find_best_zslices(channel_img)
        zslices.append(zslice)

        thresh = threshold_image(channel_img[zslice], 0.7)

        axs[0].imshow(channel_img[zslice])
        axs[1].imshow(thresh)
        axs[0].set_title("zslice")
        axs[1].set_title("threshold")

        # move the background and noise to (0, 1)
        cluster_mask = find_clusters(thresh, 100)
        
        # cluster counts
        values = set(cluster_mask.flatten())
        counts = {-2: 0, -1: 0}
        for value in values:
            count = (cluster_mask == value).sum().item()
            counts[value.item()] = count

        cluster_ids = set(cluster_mask.flatten()) - {-2, -1}
        num_clusters = len(cluster_ids)
        
        cmap = plt.get_cmap('tab10')
        norm = mcolors.Normalize(vmin=-2, vmax=cluster_mask.max())

        axs[2].imshow(cluster_mask, cmap=cmap, norm=norm)
        axs[2].set_title("clusters")
        axs[2].set_xlabel(f"num clusters={num_clusters}")

        legend_colors = [cmap(norm(val)) for val in values]
        patches = [Patch(color=color, label=f"cluster {val}: {counts[val]}") for val, color in zip(values, legend_colors)]
        axs[2].legend(handles=patches, bbox_to_anchor=(1.5,1.0))

        plt.show()


if __name__ == "__main__":
    main()