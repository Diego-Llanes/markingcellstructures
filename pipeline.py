import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

    cmap = plt.cm.get_cmap("tab10")

    # find best z_slice (might just be 24 ??)
    zslices= []
    for channel_idx, channel in enumerate(channels):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

        channel_img = img[:, channel_idx]
        zslice = find_best_zslices(channel_img)
        zslices.append(zslice)

        thresh = threshold_image(channel_img[zslice], 0.8)

        axs[0].imshow(channel_img[zslice])
        axs[1].imshow(thresh)
        axs[0].set_title("zslice")
        axs[1].set_title("threshold")
        
        cluster_mask = find_clusters(thresh, 400) + 2 # move the background and noise to positive values
        num_clusters = int(cluster_mask.max())

        axs[2].imshow(cluster_mask, cmap=cmap)
        axs[2].set_title("clusters")
        axs[2].set_xlabel(f"num clusters={int(cluster_mask.max() + 1)}")

        legend_labels = ["Background", "Noise"] + [f"Cluster {i-2}" for i in range(2, num_clusters)]

        legend_elements = [
            Patch(color=cmap(i % 10), 
            label=legend_labels[i]) for i in range(0, num_clusters)
        ]
        axs[2].legend(handles=legend_elements, bbox_to_anchor=(1.45, 1.0))

        plt.show()


if __name__ == "__main__":
    main()
