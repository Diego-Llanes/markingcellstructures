import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

from collections import namedtuple

DATA_DIR = Path(__file__).parent / 'data'
Sharpness = namedtuple(
    'Sharpness',
    ['z', 'c', 'sharpness']
)

"""
notes:
 - data is shaped as (z, c, y, x)
"""


def view_sample(
        img: np.ndarray,
        z_slices: Sharpness,
        threshold_ps: Tuple[float] = (0.81, 0.81, 0.81),
) -> None:

    # img has 3 channels, plot all
    fig, axs = plt.subplots(3, 3, figsize=(15, 8))
    channels = [
        "Cilia",
        "Golgi",
        "Cilia Base",
    ]

    for i, channel in enumerate(channels):
        axs[0][i].imshow(img[z_slices[i].z, i], cmap='gray')
        axs[0][i].set_title(f'{channel} Channel')

        # apply binary thresholding
        ret, thresh = cv2.threshold(
            img[z_slices[i].z, i],
            threshold_ps[i] * img[z_slices[i].z, i].max(),
            img[z_slices[i].z, i].max(),
            cv2.THRESH_BINARY
        )

        axs[1][i].imshow(thresh, cmap='gray')
        axs[1][i].set_title(f'{channel} Channel Thresholded to {threshold_ps[i] * 100}%')

        contours, _ = cv2.findContours(
            cv2.convertScaleAbs(thresh),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        all_points = np.concatenate(contours)
        hull = cv2.convexHull(all_points)
        cv2.drawContours(img[z_slices[i].z, i], [hull], 0, (0, 255, 0), 2)
        plt.imshow(img[z_slices[i].z, i], cmap='gray')




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

            channels[c].append(
                Sharpness(i + dist_from_center, c, sharpness)
            )

    if best_only:
        return {c: max(sharpness, key=lambda x: x.sharpness) for c, sharpness in channels.items()}
    return channels

def threshold_image(
        img: np.ndarray,
        threshold: int,
) -> np.ndarray:
    return np.where(img > threshold, 255, 0).astype(np.uint8)

def main():
    files = list(DATA_DIR.glob('*.tif'))
    sample = files[0]
    img = tifffile.imread(sample)
    z_slices = find_best_zslices(img)
    view_sample(img, z_slices)


if __name__ == "__main__":
    main()
