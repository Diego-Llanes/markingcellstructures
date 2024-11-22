import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List, Tuple, Dict
import heapq

from collections import namedtuple

# We need to integrate over the surface of the golgi apparatus
# to find the center of mass of the golgi apparatus

# Base the integration off of the shape found from convex hull over layed on top
# of the original image to find the actual center of mass

# Find the Cilia base and then if there is a Cilia within some delta distance 
# in the radius of the Cilia base, then we can say that the Cilia is attached to the Cilia base

DATA_DIR = Path(__file__).parent / "data"
Sharpness = namedtuple("Sharpness", ["z", "c", "sharpness"])

"""
notes:
 - data is shaped as (z, c, y, x)
"""

# We need a more adaptive thresholding algorithm.
def threshold_image(
    img: np.ndarray,
    z_slices: Sharpness,
    threshold_ps: Tuple[float] = (0.98, 0.98, 0.98),
    channels: List[str] = ["Cilia", "Golgi", "Cilia Base"],
) -> cv2.threshold:
    thresholded = []
    for i, channel in enumerate(channels):
        image_with_channel = img[z_slices[i].z, i]
        percentile_brightness = get_percentile_brightness(threshold_ps[i], image_with_channel)
        max_brightness = image_with_channel.max()
        
        ret, thresh = cv2.threshold(
            image_with_channel,
            percentile_brightness,
            max_brightness,
            cv2.THRESH_TOZERO,
        )
        
        # Normalize values. 
        # Go from percentile_brightness->max to 0->255
        #Then, do another thresholding to find the REALLY bright spots.
        thresh = remap_values(
            thresh,
            percentile_brightness, max_brightness,
            0, 255
        )
        
        thresh = np.uint8(thresh)
        thresh = cv2.adaptiveThreshold(
            thresh,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21, # For different zooms, this will also need to be modified
            -90 # The lower, the relatively brigher a pixel has to be to its surroundings to be included.
        )
        thresholded.append(thresh)
    return np.stack(thresholded, axis=0)

# 81% of the max brightness pixel in an image is not the 81st percentile of bright pixels.
# We need the top brightest pixels even if they are not as bright as 80% of the brightest pixel.
# this function is tremendously inefficient right now.
def get_percentile_brightness(percentile, img):
    # print("max: " + str(img.max()))
    pixel_count = int(percentile * len(img) * len(img[0]))
    heap = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            heapq.heappush(heap, img[i,j])
    return heapq.nsmallest(pixel_count, heap)[pixel_count-1] # Get the dimmest percentile brightest pixel

# Remap values of a 2d array from one range to another
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

# Visualize steps in the process of creating a final plot.
# Create the 3x3 grid of subplots, each being a specific channel and a step in the process.
# img is the raw tif file.
# z_slices are the clearest z-slices.
def demo_sample(
    img: np.ndarray,
    z_slices: Sharpness,
    threshold_ps: Tuple[float] = (0.99, 0.99, 0.99),
) -> None:
    # axs is a 2d array like an automatically scaled table.
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]

    img_copy = np.copy(img)
    
    thresh = threshold_image(img, z_slices, threshold_ps)
    contours = find_contours(thresh)
    # Set up subplots for raw image, thresholded image, and then contoured image.
    for i, channel in enumerate(channels):
        axs[0][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[0][i].set_title(f"{channel} Channel")

        axs[1][i].imshow(thresh[i], cmap="gray")
        axs[1][i].set_title(
            f"{channel} Channel Thresholded to {threshold_ps[i] * 100}%"
        )

        # 
        cv2.drawContours(img[z_slices[i].z, i], contours[i], -1, (0, 255, 0), 2)
        axs[2][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[2][i].set_title(f"{channel} Contoured")

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
            c: max(sharpness, key=lambda x: 24)
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


#Display the pixel brightnesses of an image's channel as a histogram
def show_histogram(img, channel=0):
    plt.xlabel("Pixel brightness")
    plt.ylabel("Pixel count")
    cilia_image = img[24, channel, :, :]
    max_val = np.max(cilia_image)
    plt.plot(cv2.calcHist([cilia_image], [0], None, [max_val], [0,max_val]), color="black")
    plt.show()

def main():
    files = list(DATA_DIR.glob("**/*.tif"))
    for sample in files:
        img = tifffile.imread(sample)
        print("Now looking at " + str(sample))
        z_slices = find_best_zslices(img)
        
        demo_sample(img, z_slices)
        # show_histogram(img)
    # cluster_masks = find_clusters(img, z_slices)
    # plot_image(
    #     np.stack(cluster_masks, axis=0),
    #     title="Cluster Masks",
    # )

if __name__ == "__main__":
    main()
