import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from debug import _debug_show_clusters, _debug_show_cluster_normalized, _debug_plot_histogram, _debug_show_image

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

def create_thresh_params(
    name,
    global_thresh_percentile,
    cluster_epsilon, cluster_samples,
    adaptive_block_size, adaptive_constant
):
    return {
        'channel': name,
        'global_thresh_percentile': global_thresh_percentile,
        'cluster_epsilon': cluster_epsilon,
        'cluster_samples': cluster_samples,
        'adaptive_block_size': adaptive_block_size,
        'adaptive_constant': adaptive_constant
    }
# Parameters determined:
#   Global thresholding: Threshold percentile
#   DBSCAN: Epsilon and min_samples
# Done by computing each combination and using best results of objective function.
def compute_threshold(
    channel: int,
    channel_img: np.ndarray,
    thresh_range: Tuple[float, float],
    ):
    
    # Ranges for tuning each channel.
    # These are basically magic numbers right now.
    # They weren't derived with any rigor, just "ok yeah that seems to work for most of the images"
    cilia_params = create_thresh_params(
        'cilia',
        0.95,
        25, 125,
        21, -20
    )
    golgi_params = create_thresh_params(
        'golgi',
        0.95,
        15, 300, #Golgi are more concentrated blobs compared to cilia
        65, -20
    )
    cilia_base_params = create_thresh_params(
        'cilia_base',
        0.96,
        25, 190,
        21, -10
    )
    
    channel_configuration = [cilia_params, golgi_params, cilia_base_params]
    config = channel_configuration[channel]
    print(config)
    
    #FIXME Future work? We're only looking at the first element in the range.
    # There's no point in actually doing anything iterative here because we don't have an objective function.
    for percentile in range(thresh_range[0], thresh_range[1], 1):
        percentile /= 100
        print("Testing threshold " + str(percentile))
        thresholded_img = threshold_image(channel_img, config['global_thresh_percentile'])
        
        cluster_mask = find_clusters(
            thresholded_img,
            eps=config['cluster_epsilon'],
            samples=config['cluster_samples']
        )
        
        _debug_show_clusters(cluster_mask)
        return process_clusters(channel_img, cluster_mask, config)


# Use the cluster mask to extract what was detected to be each cluster.
# After isolating each cluster, normalize the brightnesses to 0-255.
# After that, isolate bright spots.
# Debug: Shows each cluster and what the blob detector picks up.
def process_clusters(channel_img, cluster_mask, config):
    cluster_ids = sorted(list(set(cluster_mask.flatten()) - {-2, -1}))
    
    #Isolate each cluster to its own image and then threshold for the spots that appear bright.
    cluster_cilia = []
    for cluster in cluster_ids:
        # Copy channel_img values over if they are in the mask
        # Find the min and max of the values within the mask for remapping values
        mask = (cluster_mask == cluster)
        
        cluster_max = np.max(channel_img[mask]) if np.any(mask) else 0
        cluster_min = np.min(channel_img[mask]) if np.any(mask) else 0
        
        masked = np.full_like(channel_img, fill_value=cluster_min)
        masked[mask] = channel_img[mask]
        
        # The cluster points are copied, now normalize the brightnesses to be in 0-255.
        remapped = remap_values(masked, cluster_min, cluster_max, 0, 255)
        _debug_show_cluster_normalized(remapped, cluster)
        cluster_cilia.append(isolate_bright_spots(remapped, config))
    
    #Stack the masks together for the final thresholded image.
    stacked = np.zeros(dtype=np.uint8, shape=channel_img.shape)
    for cluster_cilia_mask in cluster_cilia:
        stacked = cv2.bitwise_or(stacked, cluster_cilia_mask)
    _debug_show_image(stacked, "Final step: thresholded clusters stacked together")
    return stacked

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


# Take an isolated cluster with ranges from 0-255, and isolate the bright spots
# which are hopefully at this point made up mostly of the features we want.
# Currently mostly tuned for cilia.
def isolate_bright_spots(img, config):
    img = np.uint8(img)
    adaptive_thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config['adaptive_block_size'], # May depend on zoom
        config['adaptive_constant']
    )
    _debug_show_image(adaptive_thresh, title="after adaptive thresholding")
    # Get rid of tiny artifacts for the cilia channel only
    if config['channel'] == "cilia":
        kernel_size = (3,3) #May depend on zoom
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        final_thresh = cv2.morphologyEx(
            adaptive_thresh,
            cv2.MORPH_OPEN,
            kernel,
        )
        
        _debug_show_image(final_thresh, title="after opening morph")
        return final_thresh
    return adaptive_thresh

def find_clusters(
    channel_img: np.ndarray,
    eps=25, # Epsilon should be a function of the zoom of the image as well. For now, just some number I pulled out of nowhere.
    samples=125
) -> np.ndarray:

    points = np.column_stack(np.where(channel_img > 0))

    # find clusters using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=samples) # For now, just some number I pulled out of nowhere. about this dense is good.
    labels = dbscan.fit_predict(points)

    # mark each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_mask = np.ones((1200, 1200)) * -2

    for (coord, label) in zip(points, labels):
        cluster_mask[coord[0], coord[1]] = label

    return cluster_mask

if __name__ == "__main__":
    print("Running wrong file!")