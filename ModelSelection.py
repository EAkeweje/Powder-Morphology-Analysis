import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from ShapeDescs import HuMoments
from utils import plot_sample_cluster, cluster_sizes

from cpf_core import CPFcluster


# datapath = "C:/Users/eakew/OneDrive - Trinity College Dublin/Documents/Research Work/ShapeAnalysisForMgPowder/316L-reused/316L-reused/Individual Particle Images"

def select_cpf(data, metric = 'db', n_jobs = 20):
    assert metric in ['sil', 'db', 'ch'], 'Unsupported metric'

    rhos = np.linspace(0.1, 0.9, 9)
    ks = [100, 150, 200, 300]

    cpf_grid_score = {}
    for k in tqdm(ks):
        cpf_hu = CPFcluster(k, rho = rhos, n_jobs= n_jobs)
        cpf_hu.fit(data)
        for rho in rhos:
            # if len(np.unique(cpf_hu.labels_[rho])) <= 15: #cpf giving many clusters so I need to take this off
            if metric == 'db':
                cpf_grid_score[str(k) + '_' + str(rho)] = davies_bouldin_score(data, cpf_hu.labels_[rho])
            elif metric == 'sil':
                cpf_grid_score[str(k) + '_' + str(rho)] = silhouette_score(data, cpf_hu.labels_[rho])
            elif metric == 'ch':
                cpf_grid_score[str(k) + '_' + str(rho)] = calinski_harabasz_score(data, cpf_hu.labels_[rho])

    if metric == 'db':
        k_hat, rho_hat = min(cpf_grid_score, key = cpf_grid_score.get).split('_')
    else:
        k_hat, rho_hat = max(cpf_grid_score, key = cpf_grid_score.get).split('_')

    return int(k_hat), float(rho_hat), cpf_grid_score


def k_score(data, k, metric = 'db'):
    assert metric in ['sil', 'db', 'ch'], 'Unsupported metric'
    kmeans_hm = KMeans(n_clusters = k, n_init= 'auto')
    kmeans_hm.fit(data)
    if metric == 'db':
        return davies_bouldin_score(data, kmeans_hm.labels_)
    elif metric == 'sil':
        return silhouette_score(data, kmeans_hm.labels_)
    elif metric == 'ch':
        return calinski_harabasz_score(data, kmeans_hm.labels_)



def select_kmean(data, metric = 'db', n_jobs = -1):
    with Parallel(n_jobs = n_jobs) as parallel:
        scores = parallel(delayed(k_score)(data, k, metric) for k in range(2,15))
    
    km_grid_score = {}
    for i, k in enumerate(range(2, 15)):
        km_grid_score[k] = scores[i]
    
    if metric == 'db':
        opt_k = min(km_grid_score, key = km_grid_score.get)
    else:
        opt_k = max(km_grid_score, key = km_grid_score.get)

    return opt_k, km_grid_score

def select_models(data, suffix, folders = ['kmeans', 'cpf'], metric = 'db'):
    '''
    Returns (k, k1, r) where k is optimal k for kmeans, k1 is optimal k for cpf, r is rho.
    '''
    # kmeans
    opt_k, _ = select_kmean(data, metric)
    km_model = KMeans(opt_k, n_init='auto')
    km_model.fit(data)
    if not os.path.exists(folders[0]):
        os.mkdir(folders[0])
    np.save(f'{folders[0]}/{suffix}_{metric}-labels.npy', km_model.labels_)

    # cpf
    k, r, _ = select_cpf(data, metric)
    cpf_model = CPFcluster(k, rho = r)
    cpf_model.fit(data)
    if not os.path.exists(folders[1]):
        os.mkdir(folders[1])
    np.save(f'{folders[1]}/{suffix}_{metric}-labels.npy', cpf_model.labels_)

    return opt_k, k, r