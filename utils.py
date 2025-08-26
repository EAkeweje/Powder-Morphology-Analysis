import numpy as np
import pandas as pd
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

def plot_clusterings(X, labels, label_names, method='pca', s=1, alpha=0.5,
                    random_state=42, n_neighbors=15, min_dist=0.1, reverse_order = False):
    """
    Visualize clustering using PCA or UMAP.

    Parameters:
    - X: array-like, feature matrix
    - labels: array-like, clustering labels
    - method: 'pca', 'tsne', or 'umap' for dimensionality reduction
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
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30,
                       max_iter = 1000,
                       random_state=random_state)
        reduced_X = reducer.fit_transform(X)
        x_label, y_label = 't-SNE component 1', 't-SNE component 2'
    else:
        raise ValueError("method must be 'pca', 'tsne' or 'umap'")
     
    n_clusterings = len(label_names)
    if n_clusterings == 1:
        labels = [labels]
    assert len(labels) == n_clusterings, "Number of labels must match number of clusterings"
    
    fig, ax = plt.subplots(1, n_clusterings, figsize=(5 * n_clusterings, 5))
    if n_clusterings == 1:
        ax = [ax]

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, lbls in enumerate(labels):
        ordered_lbls = np.unique(lbls) if not reverse_order else np.unique(lbls)[::-1]
        for j in ordered_lbls:
            x = reduced_X[lbls == j]
            ax[i].scatter(x[:, 0], x[:, 1], s=s, alpha=alpha, label=f'Cluster {j}',
                          color = default_colors[j % len(default_colors)])# if not reverse_order else default_colors[-j])
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel(y_label)
        ax[i].set_title(f'{label_names[i]}')
        #get handles and labels
        handles, labels = ax[i].get_legend_handles_labels()
        #specify order of items in legend
        # order = np.unique(lbls)
        # print(ordered_lbls, labels)
        #add legend to plot
        ax[i].legend([handles[idx] for idx in ordered_lbls],
                     [labels[idx] for idx in ordered_lbls]) 

    fig.tight_layout()


def plot_clustering_3d(encodings, labels):
    data = PCA(3).fit_transform(encodings)
    fig = go.Figure(data=[go.Scatter3d(x = data[:,0],
                                    y = data[:,1],
                                    z = data[:,2],
                                    mode = 'markers',
                                    marker= dict(size = 1, color = labels,
                                                 colorscale = 'Viridis'))])
    fig.show()
    return

def plot_clustering_fd(fd, labels, n):
    """
    Plot a sample of functional data objects from each cluster.
    fd : skfda.FDataGrid or array-like
        Functional data object containing the samples to be clustered.
    labels : array-like of int
        Cluster labels for each sample in `fd`.
    n : int
        Number of samples to randomly select and plot from each cluster.
    Returns
    -------
    None
        The function displays a plot of the selected samples grouped by cluster.
    """
    from skfda import FDataGrid

    fd_samples = []
    lbs = []
    for i in np.unique(labels):
        fd_i = fd[labels == i]
        idx = np.random.choice(len(fd_i), n, replace=False)
        fd_samples.append(fd_i[idx])
        lbs.extend([i] * n)
        
    FDataGrid.concatenate(*fd_samples).plot(group = lbs, legend = True,)
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

def plot_silhouette(X, labels, ax, tbox = 0.1):
    from sklearn.metrics import silhouette_samples
    
    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10  # Space between clusters
    for i in np.unique(labels):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    score = round(np.mean(silhouette_vals),  3)
    neg_count = np.where(silhouette_vals < 0)[0].shape[0] / silhouette_vals.shape[0]
    ax.text(tbox, 0, 'score: {:.2f},\nmisclass: {:.3f}'.format(score, neg_count), fontsize = 13)
    ax.axvline(np.mean(silhouette_vals), color="red", linestyle="--")  # Overall silhouette score
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot")
    ax.set_yticks([])

def morphologi_features(df, ids_to_extract):
    """
    Calculate the mean and standard deviation of morphological features for selected particle IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing particle data with columns including 'Id', 'HS Circularity', 'Convexity', and 'Aspect Ratio'.
    ids_to_extract : list
        List of particle IDs to extract features for.

    Returns
    -------
    tuple
        A tuple containing the mean and standard deviation (as pandas Series) of the selected features.
    """
    extracted_df = df[df['Id'].isin(ids_to_extract)].drop(columns=['Id'])
    return extracted_df.mean(), extracted_df.std()

def extract_ids_from_filenames(datalist, nc_lbs, cluster_label):
    """
    Extract IDs from filenames based on cluster labels.
    
    Parameters
    ----------
    datalist : list
        List of filenames.
    nc_lbs : np.ndarray
        Array of cluster labels.
    cluster_label : int
        The specific cluster label to filter by.
        
    Returns
    -------
    list
        List of extracted IDs from filenames in the specified cluster.
    """
    ids = []
    for f in np.array(datalist)[nc_lbs == cluster_label]:
        try:
            ids.append(int(os.path.splitext(f)[0].split('_')[-1]))
        except Exception as e:
            print(f"Could not extract ID from filename: {f} ({e})")
    return ids

def order_by_size_path(label_path):
    lbs = np.load(label_path, allow_pickle=True)
    # Get the unique labels and their sizes
    unique_labels, counts = np.unique(lbs, return_counts=True)
    # Sort the labels by size (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sorted_indices]
    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    # Remap the labels in the original array
    new_lbs = np.vectorize(label_mapping.get)(lbs)
    # Save the remapped labels
    new_label_path = label_path.replace('.npy', '_ordered.npy')
    np.save(new_label_path, new_lbs)
    print(f"Labels ordered by size and saved to {new_label_path}")
    return new_lbs

from typing import Optional
def order_by_size(lbs, label_path: Optional[str] = None):
    # Get the unique labels and their sizes
    unique_labels, counts = np.unique(lbs, return_counts=True)
    # Sort the labels by size (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sorted_indices]
    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    # Remap the labels in the original array
    new_lbs = np.vectorize(label_mapping.get)(lbs)
    if label_path:
        # Save the remapped labels
        np.save(label_path, new_lbs)
        print(f"Labels ordered by size and saved to {label_path}")
    return new_lbs

def plot_trad_desc(label_path, datapath, trad_desc, showfliers = True, save_plot=False):
    """
    Plot traditional morphological descriptors for each cluster using boxplots.

    Parameters:
    -----------
    label_path (str | ndarray): If str, it is path to the file containing cluster labels. Otherwise, it is a numpy array of cluster labels.
    datapath (str): Path to the directory containing powder particle images.
    trade_desc (str): path to the tab-separated file containing traditional descriptors for each particle.
    showfliers (bool): If True, shows outliers in the boxplot. Default is True.
    save_plot (bool): If True, saves the plot as a PNG file. Default is False.
    """

    if isinstance(label_path, str) and label_path.endswith('.npy'):
        labels = np.load(label_path)
    elif isinstance(label_path, np.ndarray):
        labels = label_path
    else:
        raise ValueError("label_path must be a path to a .npy file or a numpy array.")

    datalist = _getimages(datapath)

    df = pd.read_table(trad_desc, delimiter='\t', header=0, encoding='latin')
    # print("Dataframe loaded columns title:", df.columns)
    cols = ['Circularity', 'Elongation', 'Aspect Ratio']
    df = df[['Id'] + cols]  # Keep only the relevant columns

    # Prepare data for boxplots: for each cluster, collect values for each descriptor
    cluster_ids = np.unique(labels)
    boxplot_data = {col: [] for col in cols}
    for cid in cluster_ids:
        ids = extract_ids_from_filenames(datalist, labels, cid)
        cluster_df = df[df['Id'].isin(ids)]
        for col in cols:
            boxplot_data[col].append(cluster_df[col].values)

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5), sharey=False)
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].boxplot(boxplot_data[col], labels=[str(cid) for cid in cluster_ids], showfliers = showfliers)
        axes[i].set_title(col, fontsize = 15)
        axes[i].set_xlabel('Cluster label', fontsize=12)
        axes[i].set_ylabel('Value')
    plt.tight_layout()
    if save_plot:
        save_path = os.path.splitext(label_path)[0] + '_boxplot.png' if isinstance(label_path, str) else 'cluster_boxplot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()