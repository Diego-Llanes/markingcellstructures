import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List, Tuple, Dict
from collections import namedtuple

from functions import find_best_zslices, threshold_image, find_clusters

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

        axs[channel_idx][0].imshow(channel_img, cmap="gray")
        axs[channel_idx][1].imshow(filtered, cmap="gray")
        axs[channel_idx][2].imshow(tophat, cmap="gray")

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

        axs[channel_idx][0].imshow(thresh1, cmap="gray")
        axs[channel_idx][1].imshow(thresh2, cmap="gray")
        axs[channel_idx][2].imshow(thresh3, cmap="gray")

        axs[0][0].set_title("Threshold")
        axs[0][1].set_title("Mean Threshold")
        axs[0][2].set_title("Gaussian Threshold")

        axs[channel_idx][0].set_ylabel(channel)

    plt.savefig("deliverables/thresh.png", dpi=200)



def main():
    files = list(DATA_DIR.rglob("*.tif"))
    sample = [file for file in files if "_10_MMStack_Pos0.ome.tif" in str(file) and "1.5x" in str(file)][0]
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
