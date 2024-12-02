import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List, Tuple, Dict

from collections import namedtuple

Point = namedtuple('Point', ['x', 'y']) # only used for type hinting


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

    # min max normalize the image
    channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min())

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

    points = np.column_stack(
        np.where(
            channel_img > 0
        )
    )

    # find clusters using DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = dbscan.fit_predict(points)

    # mark each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_mask = np.ones((1200, 1200)) * -2

    for (coord, label) in zip(points, labels):
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
        # find the min and max points for the cluster_ids
        max_x, max_y = np.max(np.where(clusters == cluster_id), axis=1)
        min_x, min_y = np.min(np.where(clusters == cluster_id), axis=1)

        # crop the image to the cluster
        cropped_img = binary_img[min_x:max_x, min_y:max_y]

        # get the convex hull for the cropped image
        hull = generate_convex_hull(cropped_img)

        # shift the hull back to the original image
        hull = [np.array([point[0] + min_y, point[1] + min_x]) for point in hull]

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
        # get the center of mass for the hull
        hull = np.array(hull)
        center_of_mass = np.mean(hull, axis=0)
        cluster_COMs[cluster_id] = np.array(center_of_mass, dtype=int)

    return cluster_COMs


def generate_convex_hull(
    binary_img: np.ndarray, # cropped image
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


if __name__ == "__main__":
    ...
