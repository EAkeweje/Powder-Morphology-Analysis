"""
Implementation of GPmix clustering on the Centroid Distances of powder samples.

Run this script with the command:
python GPmix_clustering.py -d <path_to_dataset>
Otherwise, import the `cluster`function from this module and call it with the dataset path as a parameter.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from skfda.ml.classification import NearestCentroid

import GPmix
from GPmix.misc import hybrid_representative_selection, estimate_nclusters

from ShapeDescs import CentroidDist
from utils import plot_clustering_fd, order_by_size


def cluster(datapath):
    # Extracting CDFs from the dataset
    CDF = CentroidDist(datapath = datapath, n_points = 200, scale_by= 'max', ext = 'bmp')
    CDF.get_descs()
    cdf_features = CDF.descs
    #get representative samples
    reps_ = hybrid_representative_selection(cdf_features, .3, .05)
    print(f"Number of representative samples: {len(reps_)}")

    # Clustering the representative samples
    sm = GPmix.Smoother(basis= 'bspline')
    sm_reps = sm.fit(reps_)
    ## estimate number of clusters
    n = estimate_nclusters(sm_reps, np.arange(2,20))
    ## GPmix clustering
    proj = GPmix.Projector(basis_type= 'rl-fpc', n_proj= 12)
    coeffs = proj.fit(sm_reps)
    unigmms = GPmix.UniGaussianMixtureEnsemble(n_clusters= n, init_method= 'k-means++')
    unigmms.fit_gmms(coeffs)
    lbs = unigmms.get_clustering(weighted_sum= False)
    ## plot the clustering of representative samples
    plot_clustering_fd(sm_reps, lbs, 25)
    plt.show()

    # Full clustering
    ## Nearest Centroid classifier for full clustering
    nc_clst = NearestCentroid()
    nc_clst.fit(sm_reps, lbs)
    sm_cdfs = sm.fit(cdf_features)
    nc_lbs = nc_clst.predict(sm_cdfs)

    # Plotting the clustering of full dataset
    ## reorder the labels by size and save label
    nc_lbs = order_by_size(nc_lbs, 'GPmix_labels.npy')
    plot_clustering_fd(sm_cdfs, nc_lbs, 15)
    plt.show()

    print('DB score: ', davies_bouldin_score(cdf_features, nc_lbs),
        'CH index: ', calinski_harabasz_score(cdf_features, nc_lbs)
    )
    return nc_lbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', type=str,
                                help='Path to the dataset containing grey-level images of powder samples.')
    args = parser.parse_args()
    cluster(args.datapath)