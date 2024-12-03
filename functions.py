import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import math

from pathlib import Path
from typing import List, Tuple, Dict

from collections import namedtuple


def find_best_zslices(
    img: np.ndarray,
    best_only=True,
    dist_from_center=10,
) -> int:
    '''for a channel image [51, 1200, 1200] return the index of the best zslice'''

    z_slices = img.shape[0]
    
    best_sharpness = 0
    best_zslice_idx = 0

    start, end = (z_slices // 2) - dist_from_center, (z_slices // 2) + dist_from_center
    for i in range(start, end):
        sharpness = cv2.Laplacian(img[i], cv2.CV_64F).var()

        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_zslice_idx = i
    
    return best_zslice_idx


def threshold_image(
    channel_img: np.ndarray,
    threshold_p: float,
) -> cv2.threshold:
    ''' return a binary map threshold image'''
    percentile_brightness = get_percentile(channel_img, threshold_p)
    ret, thresh = cv2.threshold(
        channel_img,
        percentile_brightness,
        channel_img.max(),
        cv2.THRESH_BINARY,
    )
    return thresh

# Takes an image and returns the brightness of a certain percentile.
# Higher percentile, brighter pixel.
# Basically the same as using the standard deviation and mean.
def get_percentile(channel_img, percentile):
    assert percentile >= 0
    assert percentile <= 1
    
    flattened = np.array(channel_img).flatten()
    sorted_pixels = sorted(flattened, reverse=True)
    pixel_amt = math.ceil(len(flattened) * (1-percentile)) # Amount of pixels within the supplied percentile.
    return sorted_pixels[pixel_amt-1]

# Parameters determined:
#   Global thresholding: Threshold percentile
#   DBSCAN: Epsilon and min_samples
# Done by computing each combination and using best results of objective function.
def determine_best_parameters(
    channel_img: np.ndarray,
    thresh_range: Tuple[float, float],
    ):
    scores = []
    for percentile in range(thresh_range[0], thresh_range[1], 1):
        percentile /= 100
        print("Testing threshold " + str(percentile))
        thresholded_img = threshold_image(channel_img, percentile)
        cluster_mask = find_clusters(thresholded_img)
        _debug_show_clusters(cluster_mask)
        process_clusters(channel_img, cluster_mask)
        scores.append(objective(cluster_mask))
    print(scores)
    # Will be for looking at results of objective function.
    # plt.scatter(np.arange(thresh_range[0], thresh_range[1], 1), scores)
    # plt.show()

# Use the cluster mask to extract what was detected to be each cluster.
# After isolating each cluster, normalize the brightnesses to 0-255.
# run blob detection with parameters tuned for cilia (likely inertia, area, and/or threshold)
# Debug: Shows each cluster and what the blob detector picks up.
def process_clusters(channel_img, cluster_mask):
    cluster_ids = sorted(list(set(cluster_mask.flatten()) - {-2, -1}))
    print(channel_img.shape)
    
    #Isolate each cluster to its own image and then threshold for the spots that appear bright.
    for cluster in cluster_ids:
        # Copy channel_img values over if they are in the mask
        # Find the min and max of the values within the mask for remapping values
        mask = (cluster_mask == cluster)
        
        cluster_max = np.max(channel_img[mask]) if np.any(mask) else 0
        cluster_min = np.min(channel_img[mask]) if np.any(mask) else 0
        
        masked = np.full_like(channel_img, fill_value=cluster_min)
        masked[mask] = channel_img[mask]
        
        # normalize the brightnesses to be in 0-255.
        remapped = remap_values(masked, cluster_min, cluster_max, 0, 255)
        # isolate_bright_spots(remapped)
        _debug_show_cluster_normalized(remapped, cluster)
        
        

# Remap values of a 2d array from one range to another. min1 max1 to min2 max2.
# might result in floating point values?
def remap_values(img, min1, max1, min2, max2):
    range1 = max1-min1
    range2 = max2-min2
    for i in range(len(img)):
        for j in range(len(img[0])):
            val = img[i,j]
            percent = (val-min1)/range1
            remapped = min2 + (percent*range2)
            img[i,j] = remapped
    return img

def isolate_bright_spots(img):
    img = np.uint8(img)
    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        61,
        -100
    )
    plt.imshow(thresh, interpolation='none')
    plt.show()

# DEBUG. Visualize computed clusters.
def _debug_show_clusters(cluster_mask):
    cluster_ids = sorted(list(set(cluster_mask.flatten()) - {-2, -1}))
    num_clusters = len(cluster_ids)
    print("Num clusters: " + str(num_clusters))
    
    values = set(cluster_mask.flatten())
    counts = {-2: 0, -1: 0}
    for value in values:
        count = (cluster_mask == value).sum().item()
        counts[value.item()] = count
    
    # Set up visualization
    cmap = plt.get_cmap('tab20')
    norm = mcolors.Normalize(vmin=-2, vmax=cluster_mask.max())

    plt.imshow(cluster_mask, cmap=cmap, norm=norm, interpolation='none')
    plt.suptitle("clusters")
    # plt.set_xlabel(f"num clusters={num_clusters}")

    legend_colors = [cmap(norm(val)) for val in values]
    patches = [Patch(color=color, label=f"cluster {val}: {counts[val]}") for val, color in zip(values, legend_colors)]
    plt.legend(handles=patches, bbox_to_anchor=(1.85,1.0))
    plt.show()

#Used for showing each cluster isolated from everything else.
# Auto zooms to important area.
def _debug_show_cluster_normalized(masked_remapped_cluster, number):
    plt.imshow(masked_remapped_cluster, interpolation='none')
    plt.suptitle("cluster " + str(number))
    plt.show()

# Display the pixel brightnesses of an channel image as a histogram.
# Use start if you want to start from a specific brightness
def _debug_plot_histogram(channel_img, start=0):
    plt.xlabel("Pixel brightness")
    plt.ylabel("Pixel count")
    
    max_val = np.max(channel_img)
    plt.plot(cv2.calcHist([channel_img], [0], None, [max_val], [start,max_val]))
    plt.show()
    

def objective(cluster_mask):
    # FIXME
    num_clusters = cluster_mask.max() + 1
    noise_cluster_size = (cluster_mask == -1).sum()
    return num_clusters - noise_cluster_size


def find_clusters(
    channel_img: np.ndarray,
    eps=25, # Epsilon should be a function of the zoom of the image as well. For now, just some number I pulled out of nowhere.
) -> np.ndarray:

    points = np.column_stack(np.where(channel_img > 0))

    # find clusters using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=125) # For now, just some number I pulled out of nowhere. about this dense is good.
    labels = dbscan.fit_predict(points)

    # mark each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_mask = np.ones((1200, 1200)) * -2

    for (coord, label) in zip(points, labels):
        cluster_mask[coord[0], coord[1]] = label

    return cluster_mask

if __name__ == "__main__":
    print("Running wrong file!")