import tifffile
from pathlib import Path

from functions import (
    find_best_zslices,
    threshold_image,
    get_convex_hull_for_each_cluster,
    find_clusters,
)

from visualizations import plot_convex_hull, plot_hulls_of_clusters


TIF_FILE = Path("data/_1_MMStack_Pos0.ome.tif")
CHANNEL_TO_VIEW = 2
PERCENTAGE_THRESHOLD = 0.5
EPS = 100
MIN_SAMPLES = 2


def test_find_COM_for_each_cluster():
    img = tifffile.imread(TIF_FILE)
    img = img[:, CHANNEL_TO_VIEW]

    # find best zslice and threshold it to above 70%
    best_zslice = find_best_zslices(img)
    binary_img = threshold_image(img[best_zslice], PERCENTAGE_THRESHOLD)

    # find clusters
    clusters = find_clusters(
        binary_img,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    convex_hulls = get_convex_hull_for_each_cluster(
        binary_img,
        clusters,
    )


def test_find_convex_hull_per_cluster():

    img = tifffile.imread(TIF_FILE)
    img = img[:, CHANNEL_TO_VIEW]

    # find best zslice and threshold it to above 70%
    best_zslice = find_best_zslices(img)
    binary_img = threshold_image(img[best_zslice], PERCENTAGE_THRESHOLD)

    # find clusters
    clusters = find_clusters(
        binary_img,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    convex_hulls = get_convex_hull_for_each_cluster(
        binary_img,
        clusters,
    )

    # for cluster_id, hull in convex_hulls.items():
    #     plot_convex_hull(binary_img, hull)

    plot_hulls_of_clusters(binary_img, convex_hulls)

    return convex_hulls


if __name__ == "__main__":
    print("WARNING: debugging, don't run this file otherwise.")
    test_find_convex_hull_per_cluster()
