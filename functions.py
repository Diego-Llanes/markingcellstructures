import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
    flattened = np.array(channel_img).flatten()
    sorted_pixels = sorted(flattened, reverse=True)
    pixel_amt = int(len(flattened) * (1-percentile)) # Amount of pixels within the supplied percentile
    return sorted_pixels[pixel_amt-1]

def determine_best_threshold(
    channel_img: np.ndarray,
    thresh_range: Tuple[float, float],
    ):
    scores = []
    for percentile in range(thresh_range[0], thresh_range[1], 1):
        percentile /= 100
        print("Testing threshold " + str(percentile))
        thresholded_img = threshold_image(channel_img, percentile)
        cluster_mask = find_clusters(thresholded_img)
        scores.append(objective(cluster_mask))
    
    print("showing plot...")
    print(scores)
    plt.scatter(np.arange(thresh_range[0], thresh_range[1], 1), scores)
    plt.show()

def objective(cluster_mask):
    # FIXME
    breakpoint()
    num_clusters = cluster_mask.max() + 1
    noise_cluster_size = (cluster_mask == -1).sum()
    return num_clusters - noise_cluster_size


def find_clusters(
    channel_img: np.ndarray,
    eps=200,
) -> np.ndarray:

    points = np.column_stack(np.where(channel_img > 0))

    # find clusters using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(points)

    # mark each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_mask = np.ones((1200, 1200)) * -2

    for (coord, label) in zip(points, labels):
        cluster_mask[coord[0], coord[1]] = label

    return cluster_mask

if __name__ == "__main__":
    print("Running wrong file!")