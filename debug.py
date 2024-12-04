import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# DEBUG. Visualize computed clusters.
def _debug_show_clusters(cluster_mask):
    cluster_ids = sorted(list(set(cluster_mask.flatten()) - {-2, -1}))
    num_clusters = len(cluster_ids)
    print("Num clusters: " + str(num_clusters))
    
    values = set(cluster_mask.flatten())
    counts = {-2: 0, -1: 0}
    for value in values:
        count = (cluster_mask == value).sum().item()
        counts[value.item()] = count
    
    # Set up visualization
    cmap = plt.get_cmap('tab20')
    norm = mcolors.Normalize(vmin=-2, vmax=cluster_mask.max())

    plt.imshow(cluster_mask, cmap=cmap, norm=norm, interpolation='none')
    plt.suptitle("clusters")
    # plt.set_xlabel(f"num clusters={num_clusters}")

    legend_colors = [cmap(norm(val)) for val in values]
    patches = [Patch(color=color, label=f"cluster {val}: {counts[val]}") for val, color in zip(values, legend_colors)]
    plt.legend(handles=patches, bbox_to_anchor=(1.85,1.0))
    plt.show()

#Used for showing each cluster isolated from everything else.
# Auto zooms to important area.
def _debug_show_cluster_normalized(masked_remapped_cluster, number):
    _debug_show_image(masked_remapped_cluster, "cluster " + str(number))

# Generic show image.
def _debug_show_image(img, title=''):
    plt.imshow(img, interpolation='none')
    plt.suptitle(title)
    plt.show()

# Display the pixel brightnesses of an channel image as a histogram.
# Use start if you want to start from a specific brightness
def _debug_plot_histogram(channel_img, start=0):
    plt.xlabel("Pixel brightness")
    plt.ylabel("Pixel count")
    
    max_val = np.max(channel_img)
    plt.plot(cv2.calcHist([channel_img], [0], None, [max_val], [start,max_val]))
    plt.show()
    