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

from functions import find_best_zslices, threshold_image, find_clusters, compute_threshold

DATA_DIR = Path("data")

"""
notes:
 - data is shaped as (z, c, y, x)
"""

def main():
    files = list(DATA_DIR.rglob("*.tif"))
    
    for sample in files:
        print("Sampling " + str(sample))
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
        # fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
        channel = 2
        channel_img = img[:, channel]
        zslice = 24 # find_best_zslices(channel_img)
        best_slice = channel_img[zslice]
        
        thresholded = compute_threshold(channel, best_slice, (95, 96))
        # for channel_idx, channel in enumerate(channels):
            # channel_img = img[:, channel_idx]
            # zslice = 24 # find_best_zslices(channel_img)
            # best_slice = channel_img[zslice]
            
            # best_threshold = determine_best_parameters(best_slice, (95, 96))
            # print("Threshold: " + str(best_threshold))
            # thresh = threshold_image(channel_img[zslice], 0.9)


if __name__ == "__main__":
    main()
