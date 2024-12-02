import tifffile
from pathlib import Path

from functions import (
    find_best_zslices,
    threshold_image,
    get_convex_hull_for_each_cluster,
    find_clusters,
)

def test_find_convex_hull_per_cluster():
    print("WARNING: debugging, don't run this file otherwise.")
    from visualizations import plot_convex_hull

    img = tifffile.imread(Path("data/_1_MMStack_Pos0.ome.tif"))
    cillia_channel = 0
    img = img[:, cillia_channel]

    # find best zslice and threshold it to above 70%
    best_zslice = find_best_zslices(img)
    binary_img = threshold_image(
        img[best_zslice],
        0.7
    )

    # find clusters
    clusters = find_clusters(
        binary_img,
        eps=100,
        min_samples=2, # cillia can be small
    )

    convex_hulls = get_convex_hull_for_each_cluster(
        binary_img,
        clusters,
    )
    for cluster_id, hull in convex_hulls.items():
        plot_convex_hull(
            binary_img,
            hull
        )

if __name__ == "__main__":
    test_find_convex_hull_per_cluster()
