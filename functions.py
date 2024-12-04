import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
    """return a binary map threshold image"""

    # min max normalize the image
    channel_img = (channel_img - channel_img.min()) / (
        channel_img.max() - channel_img.min()
    )

    ret, thresh = cv2.threshold(
        channel_img,
        threshold_p,
        channel_img.max(),
        cv2.THRESH_BINARY,
    )
    return thresh


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

    print(pairwise_distances)

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

    import ipdb; ipdb.set_trace()
    print(channel_cluster_triplets)


if __name__ == "__main__":
    ...
