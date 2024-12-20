import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from typing import List, Dict

from functions import find_best_z_slice, threshold_image, generate_convex_hull, Point

DATA_DIR = Path("/research/jagodzinski/markingcellstructures")


def preprocessing_vis(img, channels):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
    for channel_idx, channel in enumerate(channels):
        channel_img = np.uint8(img[channel_idx])

        # apply histogram equalization and gaussian filtering
        equalized = cv2.equalizeHist(channel_img)
        filtered = cv2.GaussianBlur(equalized, (5, 5), 5)

        # apply tophat transformation
        kernel = np.ones((15, 15), np.uint8)
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)

        axs[channel_idx][0].imshow(channel_img, cmap="gray", interpolation='none')
        axs[channel_idx][1].imshow(filtered, cmap="gray", interpolation='none')
        axs[channel_idx][2].imshow(tophat, cmap="gray", interpolation='none')

        axs[0][0].set_title("Raw Cell Image")
        axs[0][1].set_title("Hist Equalized and Guassian Filtered")
        axs[0][2].set_title("Top-hat Transformation")

        axs[channel_idx][0].set_ylabel(channel)

    plt.savefig("deliverables/preprocessing.png", dpi=200)


def thresholding_vis(img, channels):
    thresh_ps = (0.3, 0.4, 0.4)

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
    for channel_idx, channel in enumerate(channels):
        channel_img = np.uint8(img[channel_idx])

        # apply histogram equalization and gaussian filtering
        equalized = cv2.equalizeHist(channel_img)
        filtered = cv2.GaussianBlur(equalized, (5, 5), 5)

        # apply tophat transformation
        kernel = np.ones((15, 15), np.uint8)
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)

        thresh1, thresh2, thresh3 = threshold_image(tophat, thresh_ps[channel_idx])

        axs[channel_idx][0].imshow(thresh1, cmap="gray", interpolation='none')
        axs[channel_idx][1].imshow(thresh2, cmap="gray", interpolation='none')
        axs[channel_idx][2].imshow(thresh3, cmap="gray", interpolation='none')

        axs[0][0].set_title("Threshold")
        axs[0][1].set_title("Mean Threshold")
        axs[0][2].set_title("Gaussian Threshold")

        axs[channel_idx][0].set_ylabel(channel)

    plt.savefig("deliverables/thresh.png", dpi=200)


def plot_convex_hull(
    binary_img: np.ndarray,
    hull: List[Point],
    show: bool = True,
) -> None:
    hull_closed = np.vstack([hull, hull[0]])
    plt.imshow(binary_img, cmap="gray", interpolation='none')
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], "r", linewidth=2)
    plt.title("Convex Hull")
    if show:
        plt.show()


def plot_hulls_of_clusters(
    img: np.ndarray,
    hulls: Dict[int, List[Point]],
    show: bool = True,
) -> None:
    plt.imshow(img, cmap="gray", interpolation='none')
    for cluster, hull in hulls.items():
        hull_closed = np.vstack([hull, hull[0]])
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], "r", linewidth=2)
    plt.title("Convex Hulls of Clusters")
    if show:
        plt.show()


def plot_COMs_of_clusters(
    img: np.ndarray,
    hulls: Dict[int, List[Point]],
    COMs: Dict[int, Point],
    show: bool = True,
) -> None:
    plt.imshow(img, cmap="gray", interpolation='none')
    for cluster, hull in hulls.items():
        hull_closed = np.vstack([hull, hull[0]])
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], "r", linewidth=2)

    for cluster, COM in COMs.items():
        plt.scatter(COM[0], COM[1], c="b", s=50)
    plt.title("Centers of Mass of Clusters")
    if show:
        plt.show()


def plot_full_image(
    img: List[np.ndarray],
    show: bool = True,
    channel_names = ["Cilia", "Golgi", "Cilia Base"],
    title: str =  None,
    gray_scale: bool = True,
) -> None:
    cmap = plt.cm.gray if gray_scale else plt.cm.tab20
    fig, ax = plt.subplots(
        1,
        len(channel_names),
        figsize=(5 * len(channel_names), 5)
    )
    if title:
        fig.suptitle(title)
    for channel_idx, channel in enumerate(channel_names):
        ax[channel_idx].imshow(img[channel_idx], cmap=cmap, interpolation='none')
        ax[channel_idx].set_title(channel)
    if show:
        plt.show()


def plot_full_image_of_hulls(
    img: list[np.ndarray],
    hulls: List[Dict[int, List[Point]]],
    show: bool = True,
    channel_names = ["Cilia", "Golgi", "Cilia Base"],
) -> None:
    fig, ax = plt.subplots(
        1,
        len(channel_names),
        figsize=(5 * len(channel_names), 5),
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Convex Hulls of Clusters")

    for channel_idx, channel_name in enumerate(channel_names):

        ax[channel_idx].imshow(img[channel_idx], cmap="gray", interpolation='none')

        for cluster, hull in hulls[channel_idx].items():
            hull_closed = np.vstack([hull, hull[0]])
            ax[channel_idx].plot(hull_closed[:, 0], hull_closed[:, 1], "r", linewidth=2)

        ax[channel_idx].set_title(channel_name)
    if show:
        plt.show()


def plot_full_image_of_clusters_and_COMs(
    img: list[np.ndarray],
    hulls: List[Dict[int, List[Point]]],
    COMs: List[Dict[int, Point]],
    show: bool = True,
    triplets: List[List[int]] = None,
    channel_names = ["Cilia", "Golgi", "Cilia Base"],
) -> None:


    fig, ax = plt.subplots(
        1,
        len(channel_names),
        figsize=(5 * len(channel_names), 5),
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Clusters and Centers of Mass")

    if triplets is not None:
        cmap = plt.cm.tab20  # Use the tab20 colormap
        colors = [cmap(i / 20) for i in range(20)]  # Generate 20 distinct colors
        triplet_colors = {tuple(triplet): colors[i % len(colors)] for i, triplet in enumerate(triplets)}
    else:
        triplet_colors = None

    for channel_idx, channel_name in enumerate(channel_names):

        ax[channel_idx].imshow(img[channel_idx], cmap="gray", interpolation='none')

        for cluster, hull in hulls[channel_idx].items():
            hull_closed = np.vstack([hull, hull[0]])

            color = "r"
            if triplets is not None:
                for triplet, triplet_color in triplet_colors.items():
                    if cluster in triplet:
                        color = triplet_color
                        break

            ax[channel_idx].plot(hull_closed[:, 0], hull_closed[:, 1], color, linewidth=2)

        for cluster, COM in COMs[channel_idx].items():
            ax[channel_idx].scatter(COM[0], COM[1], c="b", s=50)

        ax[channel_idx].set_title(channel_name)
    if show:
        plt.show()


def convex_hull_demo(
    img: np.ndarray,
) -> None:
    z_slices = find_best_z_slice(img)
    threshed_image: cv2.threshold = threshold_image(
        img=img,
        z_slices=z_slices,
        threshold_ps=(0.81, 0.81, 0.81),
        channels=["Cilia", "Golgi", "Cilia Base"],
    )
    channel = 0  # Cilia
    y_start, y_end = 1000, -1
    x_start, x_end = 600, 800

    convex_hull = generate_convex_hull(
        threshed_image[channel, y_start:y_end, x_start:x_end]
    )

    # Shift the points to the correct location
    for point in convex_hull:
        point[0] += x_start
        point[1] += y_start

    plot_convex_hull(threshed_image[channel], convex_hull)


def main():
    files = list(DATA_DIR.rglob("*.tif"))
    sample = [
        file
        for file in files
        if "_10_MMStack_Pos0.ome.tif" in str(file) and "1.5x" in str(file)
    ][0]
    sample = files[0]
    img = tifffile.imread(sample)[24]

    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]
    thresholding_vis(img, channels)


if __name__ == "__main__":
    main()
