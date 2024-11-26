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

# Thresholding is the process of separating a foreground and a background from an image, so we're able to create binary images.

# High-level: We want to generate the threshold dynamically per picture. For whatever stupid reason,
# I'd been setting it manually per attempt and was hoping some combination of local+adaptive thresholding
# given the manually set threhsold would be sufficient, when it really isn't. This algorithm will likely be novel.
# We'll need it to work for each different channel, each will likely have different characteristics because they fluoresce differently.

# First (real) attempt: We need some sort of iterative process to determine a threshold of the image.
# In our meeting we discussed a potential iterative approach. After creating some objective function, we can 
# strive to maximize this or get somewhere near the peak. This objective function can be made from results of a DBscan, essentially 
# minimizing the size of the -1 group, maximizing the # of clusters. It's hard to know exactly what was meant.
# Notes for attempt 1:
# The slips have a lot of fluorescence which appears the exact same in each slide. Is it reasonable to subtract the fluorescence that appears in all 3 slides?
# That is, subtract what's common? Cili won't appear where the dots are and where golgi are.
# The problem with this is that the base brightnesses of everything is not the same. The brightness distributions for each layer are different.
# 
# We can't always depend on the cilia being the brightest things in the image. Even if they were, some cilia would be dimmer than others.
# Can we depend on cilia to be the brightest things within their clusters? Can we do some clustering and then take the brightest blob/lines to be cilia?
# Not so confident in the current proposed method anymore. It's good that we're moving to a more algorithmic way to threshold, but
# if I'm understanding right, it still boils down to thresholding using some value, though it is determined differently.
# I feel we need something to do with the brightness to make it work properly.


# Mainly for the usage of finding percentile brightness values.
# Try not to use this function more than once for each image, because it's a little costly and the output will be the same.
# Input is a slice of an image with all of its channels.
# Output is a list with a sublist for each channel containing pixel brightnesses sorted in descending order.
def sorted_channel_brightnesses(img):
    result = {}
    for channel in range(len(img)):
        total_size = len(img[channel]) * len(img[channel,0])
        flattened = np.reshape(img[channel], shape=(total_size))
        result[channel] = sorted(flattened, reverse=True)
        
    return result

# From an already-sorted flattened list, return the value for the given percentile.
# Different from the % of the max value.
# TODO: We can probably just compute the mean and std dev for a more efficient calculation of percentiles.
def get_percentile_brightness(percentile, list):
    pixel_count = int(percentile * len(list))
    return list[pixel_count-1]

# Takes a z-slice of an image, then thresholds it based on the given threshold.
def threshold_image(
    img: np.ndarray,
    threshold,
    channels: List[str] = ["Cilia", "Golgi", "Cilia Base"],
) -> cv2.threshold:
    thresholded = []
    sorted_brightnesses = sorted_channel_brightnesses(img)
    
    for i, _ in enumerate(channels):
        image_with_channel = img[i]
        
        # It's probably more efficient to use the mean and std deviation to calculate percentiles.
        percentile_brightness = get_percentile_brightness(threshold, sorted_brightnesses[i])
        max_brightness = sorted_brightnesses[i][0]
        
        _, thresh = cv2.threshold(
            image_with_channel,
            percentile_brightness,
            max_brightness,
            cv2.THRESH_TOZERO,
        )
        
        # Normalize values. 
        # Go from percentile_brightness->max to 0->255
        #Then, do another thresholding to find the REALLY bright spots.
        # thresh = remap_values(
        #     thresh,
        #     percentile_brightness, max_brightness,
        #     0, 255
        # )

        thresholded.append(thresh)
        
    return np.stack(thresholded, axis=0)

# Try many thresholds, score them, then use the best one.
def tune_threshold(
    img: np.ndarray,
) -> None :
    
    z_slices = find_best_zslices(img)
    best = None
    scores = []
    
    # FIXME Tuning for cilia currently. Hard-coded channel 0.
    for i in range(0, 100, 1):

        threshold = i / 100
        thresholded_img  = threshold_image(img[z_slices[0].z], threshold)
        cluster = find_clusters(img[z_slices[0].z], thresholded_img, z_slices)
        score   = objective(cluster)
        pair = {"score": score, "threshold": threshold}
        
        if best is None or best[score] < pair[score]:
            best = {"score": score, "threshold": threshold}
            

#Objective function using threshold with DBSCAN that we want to maximize.
#Returns a score, which we use to inform a good enough threshold to use.
def objective(clusters):
    print("nothing here yet")

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

# Take thresholded image and draw a contour around the detected pixels.
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
) -> None:
    # axs is a 2d array like an automatically scaled table.
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]

    img_copy = np.copy(img)
    threshold = tune_threshold(img)
    thresh = threshold_image(img, z_slices, threshold)
    contours = find_contours(thresh)
    # Set up subplots for raw image, thresholded image, and then contoured image.
    for i, channel in enumerate(channels):
        axs[0][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[0][i].set_title(f"{channel} Channel")

        axs[1][i].imshow(thresh[i], cmap="gray")
        axs[1][i].set_title(
            f"{channel} Channel Thresholded to {threshold * 100}%"
        )

        cv2.drawContours(img[z_slices[i].z, i], contours[i], -1, (0, 255, 0), 2)
        axs[2][i].imshow(img[z_slices[i].z, i], cmap="gray")
        axs[2][i].set_title(f"{channel} Contoured")

    plt.show()

# Not used as far as I know. 
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

# Given an image with all slices,
# return a dictionary where each int key representing a z layer is paired with its sharpness data.
# Note: There may not be 1 *best* layer. Cilia for example sometimes are sharpest
# on different layers, such that you can't find one single layer where you can see them all.
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

# Use DBSCAN to generate cluster data for each pixel in the supplied image.
# Takes an image, best slices, and an epsilon.
# Returns a cluster mask, where each i,j pair is labeled with its cluster.
def find_clusters(
    img: np.ndarray,
    thresh,
    z_slices: Sharpness,
    eps=200,
) -> List[np.ndarray]:
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


#Display the pixel brightnesses of an image's channels as a histogram
def plot_histogram(img):
    colors = ["y","c","m"]
    plt.xlabel("Pixel brightness")
    plt.ylabel("Pixel count")
    
    focused_img = img[24]
    print(np.shape(focused_img))
    for i in range(len(focused_img)):
        channel_img = focused_img[i, :, :]
        max_val = np.max(channel_img)
        plt.plot(cv2.calcHist([channel_img], [0], None, [max_val], [0,max_val]), color=colors[i]
        )
    plt.show()

def main():
    files = list(DATA_DIR.glob("**/*.tif"))
    for sample in files:
        img = tifffile.imread(sample)
        print("Now looking at " + str(sample))
        z_slices = find_best_zslices(img)
        
        demo_sample(img, z_slices)
        print("Plotting new histogram")
        # plot_histogram(img)
        # plt.show()
    
    # cluster_masks = find_clusters(img, z_slices)
    # plot_image(
    #     np.stack(cluster_masks, axis=0),
    #     title="Cluster Masks",
    # )

if __name__ == "__main__":
    main()
