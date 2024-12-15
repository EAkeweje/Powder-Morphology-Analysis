import numpy as np
import warnings
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def _getimages(datapath, ext = '.bmp'):
    return [path for path in os.listdir(datapath) if path.endswith(ext)]

def plot_sample_cluster(datapath, labels, n, n_plots = (5, 5)):
    clstr = np.where(labels == n)[0]
    fig, ax = plt.subplots(n_plots[0], n_plots[1], figsize = (2 * n_plots[1], 2 * n_plots[0]))
    ax = ax.ravel()

    n_samples = n_plots[0] * n_plots[1]
    if len(clstr) < n_samples:
        samples = clstr
        warnings.warn(f"The cluster has only {len(clstr)} samples.")
    else:
        samples = np.random.choice(clstr, n_samples, replace= False)
    
    for i, shape_id in enumerate(samples):
        ax[i].imshow(cv2.imread(os.path.join(datapath, _getimages(datapath)[shape_id]),
                        cv2.IMREAD_GRAYSCALE), cmap = 'gray')
        ax[i].axis('off')
    fig.tight_layout()
    # plt.show()

    return

def cluster_sizes(labels):
    lst = []
    for i in np.unique(labels):
        lst.append(len(np.where(labels == i)[0]))
    return lst

def plot_clustering(X, labels, print_score = 'db', plt_fig = None):
    x_pca = PCA(2).fit_transform(X)
    for i in np.unique(labels):
        x = x_pca[labels == i]
        if plt_fig:
            plt_fig.scatter(x[:,0], x[:,1], alpha = 0.1, label = i)
            plt_fig.set_xlabel('PCA component 1')
            plt_fig.set_ylabel('PCA component 2')
            if print_score == 'db':
                plt_fig.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'DB = ' + str(round(davies_bouldin_score(X, labels), 4)))
            elif print_score == 'ch':
                plt_fig.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'CH = ' + str(round(calinski_harabasz_score(X, labels), 4)))
        else:
            plt.scatter(x[:,0], x[:,1], alpha = 0.1, label = i)
            plt.xlabel('PCA component 1')
            plt.ylabel('PCA component 2')
            if print_score == 'db':
                plt.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'DB = ' + str(round(davies_bouldin_score(X, labels), 4)))
            elif print_score == 'ch':
                plt.text
    # plt.show()
    return