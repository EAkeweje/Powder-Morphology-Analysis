"""
Run this script with either of the commands:
- python select_kmeans.py -d <path_to_dataset> -f <shape_descriptor_method> -p
- python select_kmeans.py -fa <path_to_feature_array> -p

With -p flag, the parallel compute is activated, otherwise script runs serially. The choice of command depends on whether you want to extract features from images or use a precomputed feature array.
Otherwise, import the `silhouette_analysis`function from this module and call it with a precomputed feature array
(not path to the feature array) as a parameter.)
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils import plot_silhouette, stratified_clustering_sample, plot_clusterings
from ShapeDescs import CentroidDist, ZernikeMoments, FourierDescriptor

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def silhouette_analysis(X):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Returns
    -------
    labels_list : list[np.ndarray]
        Labels for each k, ordered by ks.
    dbs : list[float]
        Davies–Bouldin scores for each k, ordered by ks.
    """
    print("Starting K-means clustering...")
    fig, ax = plt.subplots(2,5, figsize = (15,8))
    ax = ax.ravel()
    labels, dbs = [], []
    for k in range(2,12):
        clustering = KMeans(n_clusters= k, init= 'k-means++', n_init= 5).fit(X)
        lbls = clustering.labels_
        db = davies_bouldin_score(X, lbls)
        labels.append(lbls)
        dbs.append(db)
        X_strat, lb_strat = stratified_clustering_sample(X, lbls, 0.1)
        plot_silhouette(X_strat, lb_strat, ax[k-2])
    # Show the Silhoutte Plots
    plt.tight_layout()
    plt.show()

    # Visualize the clustering results
    best4 = np.argsort(dbs)[:4]
    lbls4 = [labels[i] for i in best4]
    title4 = [f'K={i+2}, DB={dbs[i]:.2f}' for i in best4]
    plot_clusterings(X, lbls4, title4, method='pca', s=1, alpha=0.5)
    plt.show()
    return lbls, dbs

def _fit_eval_one_k(
    k, X, *, n_init=5, random_state=42, sample_frac=0.1
):
    """Run KMeans for a single k and return everything needed downstream."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
    km.fit(X)
    labels = km.labels_
    db = davies_bouldin_score(X, labels)
    # Do sampling here so we don't re-run it serially later
    X_strat, lb_strat = stratified_clustering_sample(X, labels, sample_frac)
    return k, labels, db, X_strat, lb_strat

def silhouette_analysis_parallel(
    X,
    k_min = 2,
    k_max = 12,
    *,
    n_jobs=min(4, os.cpu_count() or 1),
    n_init=5,
    sample_frac=0.1,
    random_state=42
):
    """
    Parallelized version of silhouette_analysis.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature array to cluster.
    k_min : int, default=2
        Minimum number of clusters to evaluate.
    k_max : int, default=12
        One above the maximum number of clusters to evaluate (exclusive).
    n_jobs : int, default=min(4, os.cpu_count() or 1)
        Number of parallel jobs to run. Uses up to 4 or available CPUs.
    n_init : int, default=5
        Number of KMeans initializations to perform.
    sample_frac : float, default=0.1
        Fraction for stratified sampling used in silhouette plots.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    labels_list : list[np.ndarray]
        Labels for each k, ordered by ks.
    dbs : list[float]
        Davies–Bouldin scores for each k, ordered by ks.
    """
    print("Starting parallel K-means clustering...")
    ks = range(k_min, k_max + 1)
    # Compute all fits in parallel (use separate processes to avoid GIL & avoid MPL issues)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_eval_one_k)(
            k, X, n_init=n_init, random_state=random_state, sample_frac=sample_frac
        )
        for k in ks
    )

    # Sort results by k to keep order stable regardless of job completion order
    results.sort(key=lambda t: t[0])

    # Unpack
    ks_sorted = [t[0] for t in results]
    labels_list = [t[1] for t in results]
    dbs = [t[2] for t in results]
    X_sub_list = [t[3] for t in results]
    lb_sub_list = [t[4] for t in results]

    # ---- Silhouette plots (sequential) ----
    n = len(ks_sorted)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))
    # Handle case n==1 or nrows==1 gracefully
    if isinstance(ax, np.ndarray):
        axes = ax.ravel()
    else:
        axes = [ax]

    for i, (X_strat, lb_strat) in enumerate(zip(X_sub_list, lb_sub_list)):
        plot_silhouette(X_strat, lb_strat, axes[i])
        axes[i].set_title(f"K={ks_sorted[i]}")

    # Turn off any unused axes if grid > number of plots
    for j in range(len(X_sub_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Visualize best clusterings by DB index ----
    best4_idx = np.argsort(dbs)[:4]
    lbls4 = [labels_list[i] for i in best4_idx]
    titles4 = [f"K={ks_sorted[i]}, DB={dbs[i]:.2f}" for i in best4_idx]
    plot_clusterings(X, lbls4, titles4, method="pca", s=1, alpha=0.5)
    plt.show()

    return labels_list, dbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', type=str,
                                help='Path to the dataset containing grey-level images of powder samples.')
    parser.add_argument('-f', '--features', type=str, default='CDF',
                                help='Feature extraction method to use. Default is CentroidDist. To use this, ' \
                                'the datapath must be provided. Options: CDF, ZM, FD.')
    parser.add_argument('-fa', '--feature_array', type = str,
                        help = 'Path to the feature array file. If provided, it will be used instead of extracting features from images.')
    parser.add_argument('-p', action='store_true',
                        help= 'If set, run Silhouette analysis with parallel processing.')
    args = parser.parse_args()

    if args.feature_array:
        print(f"Loading features from {args.feature_array}...")
        X = np.load(args.feature_array)
    else:
        # Ensure the feature extraction method is valid
        assert args.features in ['cdf', 'zm', 'fd'], 'Invalid feature extraction method specified. Choose from: CDF, ZM, FD.'

        if args.features.lower() == 'cdf':
            print("Extracting CDF features...")
            CDF = CentroidDist(datapath=args.datapath, n_points=100, scale_by='max', ext='bmp')
            CDF.get_descs()
            X = CDF.descs
        elif args.features.lower() == 'zm':
            print("Extracting Zernike Moments features...")
            ZM = ZernikeMoments(datapath= args.datapath, degree= 5, ext= 'bmp')
            ZM.get_descs()
            X = ZM.descs
        elif args.features.lower() == 'fd':
            print("Extracting Fourier Descriptor features...")
            FD = FourierDescriptor(datapath= args.datapath, num_pairs= 5, ext= 'bmp')
            FD.get_descs()
            X = FD.descs
    
    if args.p:
        silhouette_analysis_parallel(X)
    else:
        silhouette_analysis(X)