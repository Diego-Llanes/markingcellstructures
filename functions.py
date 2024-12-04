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

Point = namedtuple("Point", ["x", "y"])  # only used for type hinting


# source: https://forum.image.sc/t/finding-the-best-focused-slice-in-a-z-stack/103401
# this is the discussion post i used for below.


# approach #1:
def calc_normalized_variance(img: np.ndarray) -> int:

    mean = np.mean(img)
    height = img.shape[0]
    width = img.shape[1]

    fi = (img - mean) ** 2
    b = np.sum(fi)

    normalized_variance = b / (height * width * mean)

    return normalized_variance


def find_best_z_slice(img: np.ndarray) -> int:

    z_slices = img.shape[0]
    best_normalized_variance = 0
    best_z_slice_idx = 0

    for i in range(z_slices):

        normalized_variance = calc_normalized_variance(img[i])

        if normalized_variance > best_normalized_variance:

            best_normalized_variance = normalized_variance
            best_z_slice_idx = i

    return best_z_slice_idx


# approach #2:
# # find_best_z_slice().
# def find_best_z_slice(
#     img: np.ndarray,
# ) -> int:

#     # number of z-slices.
#     z_slices = img.shape[0]

#     # keep track of our best score and index.
#     best_score       = 0
#     best_z_slice_idx = 0

#     # loop through all the slices.
#     for i in range(z_slices):

#         # return a 2D array of areas with rapid intensity changes.
#         laplacian = cv2.Laplacian(img[i], cv2.CV_64F)

#         # compute the mean of the absolute values of the array above.
#             # this represents the average amount of intensity change.
#         laplacian_score = np.mean(np.abs(laplacian))

#         # spread of the pixel intensity values within a slice.
#         variance = np.var(img[i])

#         # keep track of our current score.
#             # this represents the overall sharpness of the slice.
#         current_score = variance + laplacian_score

#         # if our current score is higher than the best score,
#         if current_score > best_score:

#             # keep track of it.
#             best_score       = current_score
#             best_z_slice_idx = i

#     # return the best index.
#     return best_z_slice_idx


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
def compute_best_threshold(
    channel: int,
    channel_img: np.ndarray,
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
    #fuck it we ball
    cilia_base_params = create_thresh_params(
        'cilia_base',
        0.96,
        25, 190,
        21, -10
    )
    
    channel_configuration = [cilia_params, golgi_params, cilia_base_params]
    config = channel_configuration[channel]
    
    thresholded_img = threshold_image(channel_img, config['global_thresh_percentile'])
    
    cluster_mask = find_clusters(
        thresholded_img,
        eps=config['cluster_epsilon'],
        min_samples=config['cluster_samples']
    )
    
    # _debug_show_clusters(cluster_mask)
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
        # _debug_show_cluster_normalized(remapped, cluster)
        cluster_cilia.append(isolate_bright_spots(remapped, config))
    
    #Stack the masks together for the final thresholded image.
    stacked = np.zeros(dtype=np.uint8, shape=channel_img.shape)
    for cluster_cilia_mask in cluster_cilia:
        stacked = cv2.bitwise_or(stacked, cluster_cilia_mask)
    # _debug_show_image(stacked, "Final step: thresholded clusters stacked together")
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
    # _debug_show_image(adaptive_thresh, title="after adaptive thresholding")
    # Get rid of tiny artifacts for the cilia channel only
    if config['channel'] == "cilia":
        kernel_size = (3,3) #May depend on zoom
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        final_thresh = cv2.morphologyEx(
            adaptive_thresh,
            cv2.MORPH_OPEN,
            kernel,
        )
        
        # _debug_show_image(final_thresh, title="after opening morph")
        return final_thresh
    return adaptive_thresh

def find_clusters(
    channel_img: np.ndarray,
    eps=100,
    min_samples=50,
) -> np.ndarray:

    points = np.column_stack(np.where(channel_img > 0))

    # find clusters using DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = dbscan.fit_predict(points)

    # mark each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_mask = np.ones((1200, 1200)) * -2

    for coord, label in zip(points, labels):
        cluster_mask[coord[0], coord[1]] = label

    return cluster_mask


def get_convex_hull_for_each_cluster(
    binary_img: np.ndarray,
    clusters: np.ndarray,
) -> Dict[int, List[Point]]:
    """
    take in a binary image and return a dictionary of cluster_id to convex hull
    """

    # get all unique cluster ids
    cluster_ids_and_noise = np.unique(clusters)

    # remove the noise cluster (-1)
    cluster_ids = cluster_ids_and_noise[cluster_ids_and_noise != -1]

    # remove the background cluster (-2)
    cluster_ids = cluster_ids[cluster_ids != -2]

    cluster_hulls = {}
    for cluster_id in cluster_ids:
        # crop the image to the cluster only include the cluster
        cropped_img = np.where(clusters == cluster_id, binary_img, 0)

        # get the convex hull for the cropped image
        hull = generate_convex_hull(cropped_img)

        if len(hull) != 0:
            cluster_hulls[cluster_id] = hull

    return cluster_hulls


def find_COM_for_each_cluster(
    img: np.ndarray,
    cluster_hulls: Dict[int, List[Point]],
) -> Dict[int, Point]:
    """
    Find the center of mass for each cluster
    """
    cluster_COMs = {}
    for cluster_id, hull in cluster_hulls.items():
        hull = np.array(hull)

        # find all points that fall within the hull
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 1)
        mask = mask.astype(bool)

        # create a meshgrid of x and y indices
        y_indices, x_indices = np.meshgrid(
            np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij"
        )

        # calculate the center of mass
        total_mass = np.sum(mask * img)
        com_x = np.sum(x_indices * mask * img) / total_mass
        com_y = np.sum(y_indices * mask * img) / total_mass

        cluster_COMs[cluster_id] = np.array([com_x, com_y])

    return cluster_COMs


def generate_convex_hull(
    binary_img: np.ndarray,  # cropped image
) -> List[Point]:
    """
    This will take a binary image and wrap all the points in a convex hull.
    """
    contours, _ = cv2.findContours(
        binary_img.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        print("No contours found.")
        return []

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    # Flatten the hull to (n_points, 2)
    hull = hull[:, 0, :]

    return hull


def channel_wise_cluster_alignment(
    center_of_masses: List[Dict[int, Point]],
    epsilons: float,
):
    """
    args:
        center_of_masses: List[Dict[int, Point]] - a list of 3 dictionaries
        where each dictionary is a cluster_id to center of mass
    returns:
        aligned_COMs: List[Dict[int, Point]] - a list of 3 dictionaries where
        each dictionary is a cluster_id to center of mass, where cluster_ids
        are aligned across channels. The clusters that do not have cooresponding
        clusters in other channels are removed.
    """
    """
    DS we need:
    pairwise_distances = {
        channel_id: {
            cluster_id: {
                other_channel_id: {
                    other_cluster_id: distance
                }
            }
        },
        ...
    }
    """
    # compute the pairwise distances between the center of masses for each channel and each cluster

    pairwise_distances = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for cluster_id, COM in center_of_masses[i].items():
                for other_cluster_id, other_COM in center_of_masses[j].items():
                    distance = np.linalg.norm(COM - other_COM)
                    if i not in pairwise_distances:
                        pairwise_distances[i] = {}
                    if cluster_id not in pairwise_distances[i]:
                        pairwise_distances[i][cluster_id] = {}
                    if j not in pairwise_distances[i][cluster_id]:
                        pairwise_distances[i][cluster_id][j] = {}
                    pairwise_distances[i][cluster_id][j][other_cluster_id] = distance

    # Step 2: Filter clusters to retain only those with correspondences in all three channels
    """
    each tuple in channel_cluster_triplets is of the form:
    [
            (cluster_id, cluster_id, cluster_id),
            (cluster_id, cluster_id, cluster_id),
            (cluster_id, cluster_id, cluster_id),
        ...
    ]
    and represents a triplet of clusters that have correspondences in all three channels
    so the length of channel_cluster_triplets is the number of full structures
    that we have
    """
    # Step 2: Filter clusters to retain only those with correspondences in all three channels
    channel_cluster_triplets = []
    for i in range(3):
        for cluster_id in center_of_masses[i]:
            valid_triplet = True
            matched_ids = [None, None, None]
            matched_ids[i] = cluster_id

            for other_channel in range(3):
                if other_channel == i:
                    continue

                # Find closest cluster in other_channel within epsilon radius
                closest_match = None
                min_distance = float('inf')
                for other_cluster_id, distance in pairwise_distances.get(i, {}).get(cluster_id, {}).get(other_channel, {}).items():
                    if distance <= epsilons and distance < min_distance:
                        closest_match = other_cluster_id
                        min_distance = distance

                if closest_match is None:
                    valid_triplet = False
                    break
                matched_ids[other_channel] = closest_match

            # If a valid triplet exists, add it to channel_cluster_triplets
            if valid_triplet and None not in matched_ids:
                triplet = tuple(matched_ids)
                if triplet not in channel_cluster_triplets:  # Avoid duplicates
                    channel_cluster_triplets.append(triplet)

    # Step 3: Compute the cumalative distance between the clusters in each triplet
    """
    each tuple in channel_cluster_triplets is of the form:
    [
        ((cluster_id, cluster_id, cluster_id), cum_dist),
        ((cluster_id, cluster_id, cluster_id), cum_dist),
        ((cluster_id, cluster_id, cluster_id), cum_dist),
        ...
    ]
    """
    cum_distances = []
    for triplet in channel_cluster_triplets:
        cum_dist = 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                cum_dist += pairwise_distances[i][triplet[i]][j][triplet[j]]
        cum_distances.append((triplet, cum_dist))

    # Step 4: Sort the triplets by cumulative distance
    cum_distances.sort(key=lambda x: x[1])
    seen_clusters = [
        set() for _ in range(3)
    ]
    final_triplets = []
    for triplet, dist in cum_distances:
        if not triplet[0] in seen_clusters[0] and not triplet[1] in seen_clusters[1] and not triplet[2] in seen_clusters[2]:
            seen_clusters[0].add(triplet[0])
            seen_clusters[1].add(triplet[1])
            seen_clusters[2].add(triplet[2])
            final_triplets.append(triplet)
    
    return final_triplets, seen_clusters


if __name__ == "__main__":
    ...
