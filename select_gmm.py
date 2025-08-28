"""
Run this script with either of the commands:
- python select_gmm.py -d <path_to_dataset> -f <shape_descriptor_method> -p
- python select_gmm.py -fa <path_to_feature_array> -p

With -p flag, the parallel compute is activated, otherwise script runs serially. The choice of command depends on whether you want to extract features from images or use a precomputed feature array.
Otherwise, import the `gmm_analysis` function from this module and call it with a precomputed feature array
(not path to the feature array) as a parameter.)
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_clusterings
from ShapeDescs import CentroidDist, ZernikeMoments, FourierDescriptor

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed

def gmm_analysis(X, ks = range(2,12)):
    X = X.astype(np.float64)
    if X.shape[1] > 50:
        pca = True
        X_Transfrom = PCA(n_components= 20)
        X_r = X_Transfrom.fit_transform(X)
        X_r.shape
    else:
        pca = False
    print("Starting GMM analysis...")

    BICs, AICs = [], []
    labels, dbs = [], []
    for k in ks:
        gm = GaussianMixture(k, covariance_type= 'full', n_init= 5) 
        if pca:
            gm.fit(X_r)
            BICs.append(gm.bic(X_r))
            AICs.append(gm.aic(X_r))
            lbls = gm.predict(X_r)
        else:
            gm.fit(X)
            BICs.append(gm.bic(X))
            AICs.append(gm.aic(X))
            lbls = gm.predict(X)

        db = davies_bouldin_score(X, lbls)
        labels.append(lbls)
        dbs.append(db)
    
    plt.plot(ks[1:], - np.array(BICs[1:]) + np.array(BICs[:-1]), label = '$BIC_k - BIC_{k-1}$')
    plt.plot(ks[1:], - np.array(AICs[1:]) + np.array(AICs[:-1]),  '--', label = '$AIC_k - AIC_{k-1}$')
    plt.xlabel('k')
    plt.grid()
    plt.legend()
    plt.title('$\Delta BIC_k and \Delta AIC_k$')
    plt.show()

    # Visualize the clustering results
    best4 = np.argsort(dbs)[:4]
    lbls4 = [labels[i] for i in best4]
    title4 = [f'K={i+2}, DB={dbs[i]:.2f}' for i in best4]
    plot_clusterings(X, lbls4, title4, method='pca', s=1, alpha=0.5)
    plt.show()
    return lbls, dbs



def gmm_analysis_parallel(
    X,
    k_min=2,
    k_max=11,
    n_jobs=-1,
    covariance_type='full',
    init_params="k-means++",
    n_init=5,
    pca_dim_threshold=50,
    pca_components=20,
    random_state=None,
    backend='loky',   # 'loky' (processes) for CPU-bound; 'threading' for I/O-bound
    verbose=0
):  
    X = X.astype(np.float64)
    # Decide whether to use PCA and prepare the matrix used for fitting
    if X.shape[1] > pca_dim_threshold:
        pca = True
        X_transform = PCA(n_components=pca_components, random_state=random_state)
        X_fit = X_transform.fit_transform(X)
    else:
        pca = False
        X_fit = X

    print("Starting parallel GMM analysis...")

    # Worker for a single K
    def _fit_for_k(k):
        gm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            init_params= init_params,
            n_init=n_init,
            random_state=random_state
        )
        gm.fit(X_fit)
        bic = gm.bic(X_fit)
        aic = gm.aic(X_fit)
        lbls = gm.predict(X_fit)
        # Compute Davies–Bouldin on the ORIGINAL X (matches your code)
        db = davies_bouldin_score(X, lbls)
        if verbose:
            print(f"K={k:2d} | BIC={bic:.1f} | AIC={aic:.1f} | DB={db:.4f}")
        return (k, bic, aic, lbls, db)

    # Parallel execution across K
    Ks = list(range(k_min, k_max + 1))
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_fit_for_k)(k) for k in Ks
    )

    # Sort by K and unpack
    results.sort(key=lambda t: t[0])
    ks, BICs, AICs, labels_list, dbs = map(list, zip(*results))

    # Plot BIC/AIC differences as in your original code
    if len(BICs) >= 2:
        dBIC = - np.array(BICs[1:]) + np.array(BICs[:-1])
        dAIC = - np.array(AICs[1:]) + np.array(AICs[:-1])
        # x-axis should be K from the second value onward (e.g., 3..12 in your plot)
        plt.plot(range(ks[1], ks[-1] + 1), dBIC, label='$BIC_K - BIC_{K-1}$')
        plt.plot(range(ks[1], ks[-1] + 1), dAIC, '--', label='$AIC_K - AIC_{K-1}$')
        plt.xlabel('K')
        plt.grid(True)
        plt.legend()
        plt.title('BIC and AIC differences with K')
        plt.show()

    # Visualize the best 4 clusterings by Davies–Bouldin (lower is better)
    best4_idx = np.argsort(dbs)[:4]
    lbls4 = [labels_list[i] for i in best4_idx]
    title4 = [f'K={ks[i]}, DB={dbs[i]:.2f}' for i in best4_idx]
    # Uses your existing helper
    plot_clusterings(X, lbls4, title4, method='pca', s=1, alpha=0.5)
    plt.show()

    # Return labels list (per K) and DB scores, like your intent
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
                        help= 'If set, run GMM analysis with parallel processing.')
    
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
    
    # gmm_analysis(X)
    if args.p:
        gmm_analysis_parallel(X, n_jobs= -1, verbose=1)
    else:
        gmm_analysis(X)