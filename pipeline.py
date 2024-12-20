import tifffile
import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict
from argparse import ArgumentParser

from functions import (
    find_best_z_slice,
    compute_best_threshold,
    get_convex_hull_for_each_cluster,
    find_clusters,
    find_COM_for_each_cluster,
    channel_wise_cluster_alignment,
)

from visualizations import (
    plot_hulls_of_clusters,
    plot_COMs_of_clusters,
    plot_full_image_of_clusters_and_COMs,
    plot_full_image,
    plot_full_image_of_hulls,
)

DATA_DIR = Path("/research/jagodzinski/markingcellstructures")
EPS = (3, 12, 5)
MIN_SAMPLES = (25, 70, 50)

"""
notes:
 - data is shaped as (z, c, y, x)
"""


def parse_args():
    parser = ArgumentParser(
        usage="""
python pipeline.py --image <image_name> # to process a single image
or
python pipeline.py --data_dir <path_to_data_dir> # to process all images in the given directory
"""
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help=f"Path to the directory containing the data",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Name of the data image to be processed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Name of the output directory\ndefault is 'output'",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots",
    )
    args = parser.parse_args()
    assert not (args.data_dir and args.image), "Please provide either --data_dir or --image"
    return args


def process_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    all_COMs = []
    all_hulls = []
    all_zslices = []
    all_binary_imgs = []
    all_clusters = []

    for i, channel in enumerate(["Cilia", "Golgi", "Cilia Base"]):
        print(f"Processing {channel} channel...")
        # get the channel
        channel_img = image[:, i]

        best_zslice = find_best_z_slice(channel_img)
        all_zslices.append(best_zslice)
        channel_img = channel_img[best_zslice]

        binary_img = compute_best_threshold(i, channel_img)
        all_binary_imgs.append(binary_img)

        # find clusters
        clusters = find_clusters(
            binary_img,
            eps=EPS[i],
            min_samples=MIN_SAMPLES[i],
        )
        all_clusters.append(clusters)

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

    # binary images
    plot_full_image(
        img=all_binary_imgs,
        channel_names=["Cilia", "Golgi", "Cilia Base"],
        title="Binary Images",
        show=True,
    )
    # clusters
    plot_full_image(
        img=all_clusters,
        channel_names=["Cilia", "Golgi", "Cilia Base"],
        title="Clusters in Each Channel",
        show=True,
        gray_scale=False,
    )
    # hulls
    plot_full_image_of_hulls(
        img=[image[all_zslices[i]][i] for i in range(3)],
        hulls=all_hulls,
        show=True,
    )
    # COMs
    plot_full_image_of_clusters_and_COMs(
        img=[image[all_zslices[i]][i] for i in range(3)],
        hulls=all_hulls,
        COMs=all_COMs,
        show=True,
        channel_names=["Cilia", "Golgi", "Cilia Base"],
    )

    final_triplets, good_cluster_ids = channel_wise_cluster_alignment(
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

    return final_COMS, final_hulls, all_zslices, final_triplets


def process_data(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    ...


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.image:
        image = tifffile.imread(args.image)
        final_COMS, final_hulls, final_zslices, final_triplets = process_image(image)
        img_channels: List[np.ndarray] = [image[final_zslices[i]][i] for i in range(3)]
        if args.show:
            plot_full_image_of_clusters_and_COMs(
                img=img_channels,
                hulls=final_hulls,
                COMs=final_COMS,
                triplets=final_triplets,
                channel_names=["Cilia", "Golgi", "Cilia Base"],
            )
    else:
        data_dir = Path(args.data_dir)
        raise NotImplementedError("Processing all images in a directory is not implemented yet")


if __name__ == "__main__":
    main()
