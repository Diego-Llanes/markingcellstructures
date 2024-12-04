import tifffile
from pathlib import Path

from functions import (
    find_best_z_slice,
    threshold_image,
    get_convex_hull_for_each_cluster,
    find_clusters,
    find_COM_for_each_cluster,
    channel_wise_cluster_alignment,
)

from visualizations import (
    plot_hulls_of_clusters,
    plot_COMs_of_clusters,
    plot_full_image_of_clusters_and_COMs,
)


TIF_FILE = Path("data/1-20_Stack_41_MMStack_Pos0.ome.tif")
CHANNEL_TO_VIEW = 2
PERCENTAGE_THRESHOLD = 0.4
EPS = 5
MIN_SAMPLES = 20


def test_channel_wise_cluster_alignment():
    img = tifffile.imread(TIF_FILE)

    all_COMs = []
    all_hulls = []
    all_zslices = []
    for i in range(3):
        # get the channel
        channel_img = img[:, i]

        # find best zslice and threshold it to above PERCENTAGE_THRESHOLD
        best_zslice = find_best_z_slice(channel_img)
        all_zslices.append(best_zslice)
        channel_img = channel_img[best_zslice]
        binary_img = threshold_image(channel_img, PERCENTAGE_THRESHOLD)

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
        all_hulls.append(convex_hulls)

        COMs = find_COM_for_each_cluster(
            img=channel_img,
            cluster_hulls=convex_hulls,
        )

        all_COMs.append(COMs)

    final_triplet_of_cluster_ids, good_cluster_ids = channel_wise_cluster_alignment(
        all_COMs,
        100
    )

    final_COMS = [{}, {}, {}]
    final_hulls = [{}, {}, {}]
    for i in range(3):
        channel_COMs = all_COMs[i]
        channel_hulls = all_hulls[i]

        for id, point in channel_COMs.items():
            if id in good_cluster_ids[i]:
                final_COMS[i][id] = point

        for id, hull in channel_hulls.items():
            if id in good_cluster_ids[i]:
                final_hulls[i][id] = hull

    plot_full_image_of_clusters_and_COMs(
        img=[
            img[all_zslices[0]][0],
            img[all_zslices[1]][1],
            img[all_zslices[2]][2],
        ],
        hulls=final_hulls,
        COMs=final_COMS,
        triplets=final_triplet_of_cluster_ids,
        channel_names=["Cilia", "Golgi", "Cilia Base"],
    )


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
    # test_find_COM_for_each_cluster()
    test_channel_wise_cluster_alignment()
