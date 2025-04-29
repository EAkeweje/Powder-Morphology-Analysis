import numpy as np
import warnings
import os
import cv2
import matplotlib.pyplot as plt
from plotly import graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import umap

def _getimages(datapath, ext = '.bmp'):
    return [path for path in os.listdir(datapath) if path.endswith(ext)]

def plot_powder(datapath):
    plt.imshow(cv2.imread(datapath, cv2.IMREAD_GRAYSCALE), cmap = 'gray')
    plt.axis('off')
    return

def plot_sample_cluster(datapath, labels, n, n_plots = (5, 5), ext = '.bmp'):
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
        ax[i].imshow(cv2.imread(os.path.join(datapath, _getimages(datapath, ext)[shape_id]),
                        cv2.IMREAD_GRAYSCALE), cmap = 'gray')
        ax[i].axis('off')
    fig.tight_layout()
    # plt.show()

    return

def cluster_sizes(labels, prop = False):
    lst = []
    for i in np.unique(labels):
        lst.append(len(np.where(labels == i)[0]))
    
    if prop:
        lst = np.array(lst)/np.sum(lst)
    else:
        lst = np.array(lst)
    
    return lst

# def plot_clustering(X, labels, print_score = 'db', s = 1, alpha = 0.5, plt_fig = None):
#     x_pca = PCA(2).fit_transform(X)
#     for i in np.unique(labels):
#         x = x_pca[labels == i]
#         if plt_fig:
#             plt_fig.scatter(x[:,0], x[:,1], s = s, alpha = alpha, label = i)
#             plt_fig.set_xlabel('PCA component 1')
#             plt_fig.set_ylabel('PCA component 2')
#             if print_score == 'db':
#                 plt_fig.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'DB = ' + str(round(davies_bouldin_score(X, labels), 4)))
#             elif print_score == 'ch':
#                 plt_fig.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'CH = ' + str(round(calinski_harabasz_score(X, labels), 4)))
#         else:
#             plt.scatter(x[:,0], x[:,1], s = s, alpha = alpha, label = i)
#             plt.xlabel('PCA component 1')
#             plt.ylabel('PCA component 2')
#             if print_score == 'db':
#                 plt.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'DB = ' + str(round(davies_bouldin_score(X, labels), 4)))
#             elif print_score == 'ch':
#                 plt.text(x = np.mean(x_pca[:,0]), y = min(x_pca[:,1]), s = 'CH = ' + str(round(calinski_harabasz_score(X, labels), 4)))
#     # plt.show()
#     return

def plot_clustering(X, labels, method='pca', print_score='db', s=1, alpha=0.5, plt_fig=None,
                    random_state=42, n_neighbors=15, min_dist=0.1):
    """
    Visualize clustering using PCA or UMAP.

    Parameters:
    - X: array-like, feature matrix
    - labels: array-like, clustering labels
    - method: 'pca' or 'umap' for dimensionality reduction
    - print_score: 'db' or 'ch' to print clustering evaluation metric
    - s: point size in scatter plot
    - alpha: transparency
    - plt_fig: optional matplotlib axis to plot into
    - random_state: random seed for UMAP
    - n_neighbors: UMAP parameter
    - min_dist: UMAP parameter
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_X = reducer.fit_transform(X)
        x_label, y_label = 'PCA component 1', 'PCA component 2'
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=random_state,
                            n_neighbors=n_neighbors, min_dist=min_dist)
        reduced_X = reducer.fit_transform(X)
        x_label, y_label = 'UMAP component 1', 'UMAP component 2'
    else:
        raise ValueError("method must be 'pca' or 'umap'")

    for i in np.unique(labels):
        x = reduced_X[labels == i]
        if plt_fig:
            plt_fig.scatter(x[:, 0], x[:, 1], s=s, alpha=alpha, label=i)
        else:
            plt.scatter(x[:, 0], x[:, 1], s=s, alpha=alpha, label=i)

    if plt_fig:
        plt_fig.set_xlabel(x_label)
        plt_fig.set_ylabel(y_label)
    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    if print_score == 'db':
        score = davies_bouldin_score(X, labels)
        score_text = f'DB = {score:.4f}'
    elif print_score == 'ch':
        score = calinski_harabasz_score(X, labels)
        score_text = f'CH = {score:.4f}'
    else:
        score_text = None

    if score_text:
        if plt_fig:
            plt_fig.text(x=np.mean(reduced_X[:, 0]), y=min(reduced_X[:, 1]), s=score_text)
        else:
            plt.text(x=np.mean(reduced_X[:, 0]), y=min(reduced_X[:, 1]), s=score_text)

    return


def plot_clustering_3d(encodings, labels):
    data = PCA(3).fit_transform(encodings)
    fig = go.Figure(data=[go.Scatter3d(x = data[:,0],
                                    y = data[:,1],
                                    z = data[:,2],
                                    mode = 'markers',
                                    marker= dict(size = 1, color = labels, colorscale = 'Viridis'))])
    fig.show()
    return

def stratified_clustering_sample(X: list, y: list, prop: float):
    sample = []
    sample_lbs = []
    for i in np.unique(y):
        clster = X[np.where(y == i)]
        n_clster = len(clster)
        sample.append(clster[np.random.choice(n_clster, int(prop * n_clster), replace= False)])
        sample_lbs += int(prop * n_clster) * [i]

    return np.concatenate(sample), sample_lbs