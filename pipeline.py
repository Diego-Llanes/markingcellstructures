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

from functions import find_best_zslices, threshold_image, find_clusters, determine_best_parameters

DATA_DIR = Path("data")

"""
notes:
 - data is shaped as (z, c, y, x)
"""

# Second proposed approach:
# Two main parts of the algorithm. First, a rudimentary thresholding to get rid of data that is definitely useless.
# We do our parameter looping thing on the following pipeline:
# Clustering (DBSCAN most likely)
# For each cluster found, do cv2 blob detection on each cluster found independently from each other.
    # Probably normalize pixel brightness belonging to each cluster.
# This blob detection should find the cilia blobs.
# This will isolate our bright cilia blobs from everything else.
# Parameter looping alg on blob detection? or clustering parameters?

#Unknown how well it will work for golgi and cilia base.
# It's crucial to produce visualizations of each step in the process for debugging purposes.

# Rationale: Our previous methods are too simple. Global doesn't work because of differing cilia brightnesses.
# Adaptive didn't quite work either. When going from relatively dark to relatively bright,
# we get a lot more than just the cilia. We also get the border of cells.
# To remedy these two issues, two different steps. 
#   1. Rudimentary global thresholding. This will get rid of data that would be most definitely useless to us. 
#   Darkness. We create areas of localized brightness with the rest being completely dark.
#   We choose a conservative global threshold like 85-90%, so it would take a relatively huge blob of noise to ruin the algorithm.
# At this point we still definitely have the cilia, and we've removed most of the irrelevant parts of the image.
# DBSCAN should have no problems making groupings because aren't particularly interested in the specialized parameters for each channel here. We only want to 
# broadly cluster the regions so we can hopefully separate cilia from each other and operate on each cluster independently.
#   2. DBSCAN the remaining image. This will produce our cluster mask.
# Now we have our independent clusters. Each cluster will be narrowed down to a cilia (or multiple?) later.
#   3. For each cluster, we want to normalize the values to a decent range, ignoring noise cluster and background cluster.
#   4. Now with our normalized cluster, we will select everything that is relatively bright and long compared to the rest of the cluster.
# Cilia are the main features (if they are there). If there are no cilia, there shouldn't be anything matching a description like this
# It's very likely that non-cilia are detected, but these will be filtered out when we don't find one of the other necessary cell structures nearby.

# Progress:
# 1 - done
# 2 - We should maximize an objective function while changing the parameters listed below. 
#   Changing these parameters affects how groups form which could be significant.
#   - done
# 3 - BLOB DETECTION IS NOT SUITABLE. Blob detection works when the blobs have the exact same color throughout.
#   - We want to isolate long thin blobs of relatively high brightness throughout.
#       - We shouldn't threshold by percentile anymore at this point. At this point, the problem is a lot easier to solve.

#Thing we might want to loop over for objective function:
#   - Epsilon/min samples for DBSCAN.
#       - Different preset ranges for each channel.
#   - 1st step global thresholding percentile value

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
        channel_img = img[:, 0]
        zslice = 24 # find_best_zslices(channel_img)
        best_slice = channel_img[zslice]
        
        threshold = determine_best_parameters(best_slice, (95, 96))
        # for channel_idx, channel in enumerate(channels):
            # channel_img = img[:, channel_idx]
            # zslice = 24 # find_best_zslices(channel_img)
            # best_slice = channel_img[zslice]
            
            # best_threshold = determine_best_parameters(best_slice, (95, 96))
            # print("Threshold: " + str(best_threshold))
            # thresh = threshold_image(channel_img[zslice], 0.9)


if __name__ == "__main__":
    main()
