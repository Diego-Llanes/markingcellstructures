import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List, Tuple, Dict

from collections import namedtuple

DATA_DIR = Path(__file__).parent / "data"
Sharpness = namedtuple("Sharpness", ["z", "c", "sharpness"])

"""
notes:
 - data is shaped as (z, c, y, x)
"""


def threshold_image(
    img: np.ndarray,
    z_slices: Sharpness,
    threshold_ps: Tuple[float] = (0.81, 0.81, 0.81),
    channels: List[str] = ["Cilia", "Golgi", "Cilia Base"],
) -> cv2.threshold:
    threses = []
    for i, channel in enumerate(channels):
        ret, thresh = cv2.threshold(
            img[z_slices[i].z, i],
            threshold_ps[i] * img[z_slices[i].z, i].max(),
            img[z_slices[i].z, i].max(),
            cv2.THRESH_BINARY,
        )
        threses.append(thresh)
    return np.stack(threses, axis=0)


def find_contours(
    thresh: np.ndarray,
) -> None:
    contours = []
    for i in range(thresh.shape[0]):
        contours.append(
            cv2.findContours(
                cv2.convertScaleAbs(thresh[i]),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
        )
    return contours


def demo_sample(
    img: np.ndarray,
    z_slices: Sharpness,
    threshold_ps: Tuple[float] = (0.81, 0.81, 0.81),
) -> None:

    # img has 3 channels, plot all
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]

    thresh = threshold_image(img, z_slices, threshold_ps)
    contours = find_contours(thresh)
    for i, channel in enumerate(channels):
        axs[0][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[0][i].set_title(f"{channel} Channel")

        axs[1][i].imshow(thresh[i], cmap="gray")
        axs[1][i].set_title(
            f"{channel} Channel Thresholded to {threshold_ps[i] * 100}%"
        )

        cv2.drawContours(img[z_slices[i].z, i], contours[i], -1, (0, 255, 0), 2)
        axs[2][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[2][i].set_title(f"{channel} Channel Contours")

    plt.show()


def plot_image(img: np.ndarray, title: str = "Image", show=True) -> None:
    if len(img.shape) > 2:
        fig, axs = plt.subplots(1, img.shape[0], figsize=(8, 8))
        for i in range(img.shape[0]):
            axs[i].imshow(img[i], cmap="gray")
            axs[i].set_title(f"Channel {i}")
    else:
        plt.imshow(img, cmap="gray")
        plt.title(title)

    if show:
        plt.show()


def find_best_zslices(
    img: np.ndarray,
    best_only=True,
    dist_from_center=10,
) -> Dict[int, Sharpness]:

    channels = {}
    z_slices = img.shape[0]
    start, end = (z_slices // 2) - dist_from_center, (z_slices // 2) + dist_from_center
    for i in range(start, end):
        for c in range(img.shape[1]):
            sharpness = cv2.Laplacian(img[i, c], cv2.CV_64F).var()

            if channels.get(c) is None:
                channels[c] = []

            channels[c].append(Sharpness(i + dist_from_center, c, sharpness))

    if best_only:
        return {
            c: max(sharpness, key=lambda x: x.sharpness)
            for c, sharpness in channels.items()
        }
    return channels


def find_clusters(
    img: np.ndarray,
    z_slices: Sharpness,
    eps=200,
) -> List[np.ndarray]:
    thresh = threshold_image(img, z_slices)
    contours = find_contours(thresh)

    cluster_masks = []

    for channel_idx, channel_contours in enumerate(contours):
        points = []
        for contour in channel_contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                # find the centroid of the contour if the area is not zero
                # "m00" is the area of the contour
                # "m10" is the sum of the X coordinates of the contour
                # "m01" is the sum of the Y coordinates of the contour
                cx = int(moments["m10"] / moments["m00"])  # X of centroid
                cy = int(moments["m01"] / moments["m00"])  # Y of centroid
                points.append([cx, cy])

        if not points:
            cluster_masks.append(
                np.zeros_like(img[z_slices[channel_idx].z, channel_idx], dtype=np.uint8)
            )
            continue

        points = np.array(points)

        # find clusters using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(points)

        # mark each cluster
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_mask = np.zeros_like(
            img[z_slices[channel_idx].z, channel_idx], dtype=np.uint8
        )

        # FIXME: The shapes drawn by fillPoly are super janky
        # Once we find the clusters, we should draw the shapes
        # using convex hulls or something similar
        for cluster_id in range(num_clusters):
            mask = np.zeros_like(cluster_mask)
            cluster_points = points[labels == cluster_id]
            cv2.fillPoly(mask, [np.array(cluster_points)], 255)
            cluster_mask = cv2.bitwise_or(cluster_mask, mask)

        cluster_masks.append(cluster_mask)

    return cluster_masks


def main():
    files = list(DATA_DIR.glob("*.tif"))
    sample = files[0]
    img = tifffile.imread(sample)
    z_slices = find_best_zslices(img)
    # demo_sample(img, z_slices)
    cluster_masks = find_clusters(img, z_slices)
    plot_image(
        np.stack(cluster_masks, axis=0),
        title="Cluster Masks",
    )


if __name__ == "__main__":
    main()
