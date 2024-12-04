import tifffile
from pathlib import Path

from functions import (
    find_best_z_slice,
    threshold_image,
    get_convex_hull_for_each_cluster,
    find_clusters,
    find_COM_for_each_cluster,
)

from visualizations import (
    plot_hulls_of_clusters,
    plot_COMs_of_clusters,
)


TIF_FILE = Path("data/_1_MMStack_Pos0.ome.tif")
CHANNEL_TO_VIEW = 2
PERCENTAGE_THRESHOLD = 0.3
EPS = 5
MIN_SAMPLES = 20


def test_find_COM_for_each_cluster():
    img = tifffile.imread(TIF_FILE)
    img = img[:, CHANNEL_TO_VIEW]

    best_zslice = find_best_z_slice(img)
    img = img[best_zslice]
    binary_img = threshold_image(img, PERCENTAGE_THRESHOLD)

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

    # integrate the convex hulls on the full florecent image
    COMs = find_COM_for_each_cluster(
        img=img,
        cluster_hulls=convex_hulls,
    )

    plot_COMs_of_clusters(
        img=img,
        hulls=convex_hulls,
        COMs=COMs,
    )


def test_find_convex_hull_per_cluster():

    img = tifffile.imread(TIF_FILE)
    img = img[:, CHANNEL_TO_VIEW]

    # find best zslice and threshold it to above 70%
    best_zslice = find_best_z_slice(img)
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

    plot_hulls_of_clusters(
        img[best_zslice],
        convex_hulls
    )


if __name__ == "__main__":
    print("WARNING: debugging, don't run this file otherwise.")
    # test_find_convex_hull_per_cluster()
    test_find_COM_for_each_cluster()
