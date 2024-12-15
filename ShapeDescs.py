import os
import imutils
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed

from cpf_core import CPFcluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
import math
from skimage.measure import *
from scipy.fft import fft
from skimage.transform import radon
from scipy.cluster.vq import kmeans,vq
from pyefd import elliptic_fourier_descriptors
from sklearn.cluster import *
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FastICA, PCA
from scipy.optimize import brent
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, cosine


def _getimages(datapath, ext = '.bmp'):
    return [path for path in os.listdir(datapath) if path.endswith(ext)]

# Hu moments
class HuMoments():
    '''
    datapath : string
        path of directory where the image data is stored.
    '''
    def __init__(self, datapath) -> None:
        self.datapath = datapath
        pass

    def hu(self, image):
        imgpath = os.path.join(self.datapath, image)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        moments = cv2.moments(img)
        return cv2.HuMoments(moments).flatten()
    
    def get_decs(self):
        with Parallel(n_jobs= -1) as parallel:
            self.descs = np.array(parallel(delayed(self.hu)(i) for i in _getimages(self.datapath)))
        pass

# Zernike moments
class ZernikeMoments():
    '''
    datapath : string
        path of directory where the image data is stored.
    Degree : integer, optional
        Maximum degree to use (default: 8)
    '''
    def __init__(self, datapath, degree= 8) -> None:
        self.datapath = datapath
        self.degree = degree
        pass


    def scale_contour(self, cnt, scale):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = (cnt_norm * scale).astype('int')
        cnt_scaled = cnt_scaled + [cx, cy]
        return cnt_scaled.astype(np.int32)

    def translate_contour(self, cnts, cx, cy):
        M = cv2.moments(cnts)
        if M['m00'] != 0:
            cxcy = np.array([M['m10']/M['m00'], M['m01']/M['m00']]) - np.array([cx, cy])
        else:
            cxcy = np.zeros(2)
        return (cnts - cxcy).astype('int32')

    def descriptors(self, img):
        img = cv2.bitwise_not(img)
        w, h = img.shape
        img_size = 200
        scale = (img_size-20) / max([w,h])
        outline = np.zeros((img_size, img_size), dtype = "uint8")
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        cnts = self.scale_contour(cnts, scale)
        cnts = self.translate_contour(cnts, img_size//2, img_size//2)
        zm = mahotas.features.zernike_moments(cv2.drawContours(outline, [cnts], 0, 255, -1), img_size//2, self.degree)
        return zm

    def _get_descriptors(self, img_id):
        imgpath = os.path.join(self.datapath, img_id)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        thresh, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return self.descriptors(img)
    
    def get_descs(self):
        with Parallel(n_jobs= -1) as parallel:
            self.descs = np.array(parallel(delayed(self._get_descriptors)(i) for i in _getimages(self.datapath)))
        pass

# Shift Invariant Feature Transformation
class SIFT():
    '''
    datapath : string
        path of directory where the image data is stored.
    k : integer, optional
        Size of global SIFT descriptor (default: 16)
    kmeans_iter : integer, optional
        Number of iterations for kmeans used for clustering local SIFT desicriptors (default: 10).
    '''
    def __init__(self, datapath, k = 16, kmeans_iter = 10) -> None:
        self.datapath = datapath
        self.k = k
        self.km_iter = kmeans_iter
        pass

    def sift_keypoints(self, img_name, n_keys = 0):
        #read image
        imgpath = os.path.join(self.datapath, img_name)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        # compute sift
        _, desc = cv2.SIFT_create(n_keys).detectAndCompute(img, None)
        return desc

    def get_descs(self):
        with Parallel(n_jobs= -1) as parallel:
            #get code book
            sift_desc_all = parallel(delayed(self.sift_keypoints)(i)
                                                        for i in tqdm(_getimages(self.datapath), desc= 'Computing local descriptors'))
            sift_desc = list(itertools.filterfalse(lambda item: type(item) != np.ndarray, sift_desc_all))
            sift_desc = np.concatenate(sift_desc)
            self.code, _ = kmeans(sift_desc, self.k, iter= self.km_iter)

        self.descs = np.zeros((len(_getimages(self.datapath)), self.k), "int16")
        for i, kps_desc in enumerate(tqdm(sift_desc_all, desc= 'Encoding local keypoint descriptors to form global descriptors')):
            if type(kps_desc) is np.ndarray:
                kp_cluster, _ = vq(kps_desc, self.code)

                for j in kp_cluster:
                    self.descs[i, j] += 1
        pass

# Elliptic Fourier Descriptors
class EllipticFourierDesc():
    '''
    datapath : string
        path of directory where the image data is stored.
    order : integer, optional
        The order of Fourier coefficients to calculate (default: 25)
    '''
    def __init__(self, datapath, order = 25) -> None:
        self.datapath = datapath
        self.order = order
        pass

    def get_efd(self, img):
        ## get img contour
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze()
        
        #efd
        ## Normalize the contour and compute the Fourier descriptors
        coeffs = elliptic_fourier_descriptors(contour, order= self.order, normalize=True)
        ## first 3 coefficients are 1, 0, and 0 snce the coefficients are normalize. So they can be dropped
        return coeffs.flatten()[3:]
    
    def _shape_efd(self, path):
        imgpath = os.path.join(self.datapath, path)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        return self.get_efd(img)
    
    def get_descs(self):
        with Parallel(n_jobs= -1) as parallel:
            self.descs = np.array(parallel(delayed(self._shape_efd)(i) for i in _getimages(self.datapath)))
        pass

# Fourier Descriptors
class FourierDescriptor():
    '''
    datapath : string
        path of directory where the image data is stored.
    num_pairs : integer, optional
        The number of pair of Fourier coefficients to calculate (default: 20)
    '''
    def __init__(self, datapath, num_pairs = 20) -> None:
        self.datapath = datapath
        self.num_pairs = num_pairs

    def get_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = max(contours, key=cv2.contourArea)
        contour_array = contour[:, 0, :]
        return contour_array
    
    def get_descriptors(self, contour_array):
        complex_contour = np.array(contour_array[:, 0] + 1j * contour_array[:, 1])
        return fft(complex_contour)
    
    def get_trans_invariant(self, fourier_desc):
        len_fd = len(fourier_desc)
        if self.num_pairs <= len_fd//2:
            array = np.concatenate([fourier_desc[1: self.num_pairs+1], fourier_desc[-self.num_pairs:]])
        else:
            array = np.zeros(2 * self.num_pairs, dtype= np.complex64)
            array[:len_fd//2] = fourier_desc[1: len_fd//2 +1]
            if len_fd % 2 == 1:
                array[-(len_fd//2) :] = fourier_desc[len_fd//2 +1:]
            else:
                array[-(len_fd//2 - 1) :] = fourier_desc[len_fd//2 +1:]
        return array
    
    def get_scale_invariant(self, array):
        return array / (np.power(np.abs(array), 2).sum() ** 0.5)
    
    def start_point_phase(self, phase, fourier_desc):
        sum = 0
        num_pair = len(fourier_desc) //2
        for i in range(num_pair):
            fd1 = fourier_desc[- 1 - i] * np.exp(-1j * (i+1) * phase)
            fd2 = fourier_desc[i] * np.exp(1j * (i+1) * phase)
            sum += fd1.real * fd2.imag - fd2.real * fd1.imag
        return sum

    def get_rotation_invariant(self, array):
        d_fun = lambda x: -self.start_point_phase(x, array)
        phi = brent(d_fun, brack = (0, np.pi)) % np.pi
        sp_inv_a = array * np.exp(1j* np.concatenate([np.arange(1,(self.num_pairs+1)), np.arange(-self.num_pairs,0)]) * phi)
        sp_inv_b = array * np.exp(1j* np.concatenate([np.arange(1,(self.num_pairs+1)), np.arange(-self.num_pairs,0)]) * (phi + np.pi))

        fd_sum_a = np.sum((1 / np.arange(1,(self.num_pairs+1))) * (sp_inv_a[:self.num_pairs] + sp_inv_a[self.num_pairs:][::-1]))
        beta_a = np.arctan(fd_sum_a.imag/fd_sum_a.real)
        fd_sum_b = np.sum((1 / np.arange(1,(self.num_pairs+1))) * (sp_inv_b[:self.num_pairs] + sp_inv_b[self.num_pairs:][::-1]))
        beta_b = np.arctan(fd_sum_b.imag/fd_sum_b.real)

        rot_inv_a = sp_inv_a * np.exp(-1j * beta_a)
        rot_inv_b = sp_inv_b * np.exp(-1j * beta_b)
        return rot_inv_a, rot_inv_b
    
    def compute_desc(self, img_id, rot_invariant = False, r_only = False):
        imgpath = os.path.join(self.datapath, img_id)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)

        contour = self.get_contour(img)
        desc = self.get_descriptors(contour)
        desc = self.get_trans_invariant(desc)
        desc = self.get_scale_invariant(desc)

        if rot_invariant:
            desc = self.get_rotation_invariant(desc)
            if r_only:
                desc = np.abs(desc[0]) #both have same magnitude, so one is enough
        else:
            if r_only:
                desc = np.abs(desc)

        return desc
    
    def get_descs(self):
        with Parallel(n_jobs = -1) as parallel:
            self.descs = np.array(parallel(delayed(self.compute_desc)(i, False, True) for i in _getimages(self.datapath)))
        pass

# Shape Context
class ShapeContext(object):
    '''
    datapath : string
        path of directory where the image data is stored.
    ...
    '''
    def __init__(self, datapath, n_contour_points = 100, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        self.datapath = datapath
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.n_contour_points = n_contour_points

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            This algorithm has dificulty O(n^3)
            return total modification cost, indexes of matched points
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def get_contour(self, img, n_points = None, scale = False):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = max(contours, key=cv2.contourArea)
        contour_array = contour[:, 0, :]

        if scale:
            contour_array = self._scale_contour(cv2.UMat(contour_array))

        if n_points:
            contour_array = self._get_contour_points(contour_array, n_points)
        return contour_array
    
    def _get_contour_points(self, contour, n_points):
        pts = np.linspace(0, len(contour)-1, n_points).astype(np.int32)
        contour_red = contour.squeeze()[pts]
        return contour_red
    
    def _scale_contour(contour, scale = None):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
        else:
            cx, cy = 0, 0

        cnt_norm = contour - [cx, cy]
        if not scale:
            dist = cdist(contour, np.array([[cx,cy]]))
            print(dist.max())
            scale = 1 / dist.max()
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        return cnt_scaled.astype(np.int32)

    def _cost(self, hi, hj):
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])
        return cost * 0.5
    
    def get_cosine_dist_matrix(self, P, Q):
        """
            Recommended as a faster alternative to chi-squared
        """ 
        P_row_sum = P.sum(axis = 1)[:, np.newaxis]
        Q_row_sum = Q.sum(axis = 1)[:, np.newaxis]
        return cdist(P/P_row_sum, Q/Q_row_sum, 'cosine')
    
    def get_cost_matrix(self, P, Q):
        P_row_sum = P.sum(axis = 1)[:, np.newaxis]
        Q_row_sum = Q.sum(axis = 1)[:, np.newaxis]
        return cdist(P/P_row_sum, Q/Q_row_sum,  self._cost)

    def compute_description(self, img_id):
        """
          Here we are computing shape context descriptor
        """
        imgpath = os.path.join(self.datapath, img_id)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)

        contour_points = self.get_contour(img, self.n_contour_points)
        t_contour_points = len(contour_points)
        # getting euclidian distance
        r_array = cdist(contour_points, contour_points)
        # getting two contour_points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        max_points = np.unravel_index(r_array.argmax(), r_array.shape)
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_bins = np.concatenate([np.zeros(1), r_bin_edges])

        # getting angles in radians
        theta_array = cdist(contour_points, contour_points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_contour_points, t_contour_points)) - np.identity(t_contour_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0
        # angles in [0,2Pi)
        theta_array_2 = theta_array % (2*math.pi)
        theta_bins = np.linspace(0, 2*math.pi, self.nbins_theta + 1)

        #take off the diagonal elements which are redundant
        r_array_n = r_array_n[~np.eye(len(r_array_n), dtype= bool)].reshape(len(r_array_n), -1)
        theta_array_2 = theta_array_2[~np.eye(len(theta_array_2), dtype= bool)].reshape(len(theta_array_2), -1)

        # building point descriptor based on angle and distance
        descriptor = []
        for i in range(t_contour_points):
            local_desc, self.r_bin_edges, self.theta_bin_edges = np.histogram2d(r_array_n[i], theta_array_2[i], [r_bins, theta_bins])
            descriptor.append(local_desc.flatten())
        return np.array(descriptor)

    def get_matching_cost(self, P, Q):
        C = self.get_cost_matrix(P, Q)
        cost, indexes = self._hungarian(C)
        return cost, indexes
    
    def get_fast_matching_cost(self, P, Q):
        C = self.get_cosine_dist_matrix(P, Q)
        cost, indexes = self._hungarian(C)
        return cost, indexes
    
    def get_descs(self):
        with Parallel(n_jobs = 5) as parallel:
            self.descs = np.array(parallel(delayed(self.compute_description)(i) for i in _getimages(self.datapath)[:5]))

            # the implementation of matching cost is not included here.
        pass


# Centroid Distance function
class CentroidDist():
    '''
    datapath : string
        path of directory where the image data is stored.
    n_points : integer, optional
        number of points to sample on image contour (default: 100)
    scale_by : 'max' or 'avg'
        how transform featured to achieve scale invariance. 'max' for scaling with maximum radius. 'avg' for
        scaling with average radius.
    '''
    def __init__(self, datapath, n_points = 100, scale_by = 'max') -> None:
        self.datapath = datapath
        self.n_points = n_points
        self.scale_by = scale_by
        pass

    def cdf(self, img_id):
        imgpath = os.path.join(self.datapath, img_id)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = max(contours, key=cv2.contourArea)
        contour_array = contour[:, 0, :]

        moments = cv2.moments(img)
        x, y = moments['m10']/ moments['m00'], moments['m01']/moments['m00']
        dist = cdist(contour_array, np.array([[x,y]]))

        if self.scale_by == 'max':
            dist /= np.max(dist)
        elif self.scale_by == 'avg':
            dist /= np.mean(dist)
        
        max_ind = dist.argmax()
        dist = np.concatenate([dist[max_ind:], dist[:max_ind]]) #shift curve such that the farthest point becomes the starting point
        dist = dist[np.linspace(0, dist.shape[0] - 1, self.n_points).astype(int)]

        return dist
    
    def get_descs(self):
        with Parallel(n_jobs = -1) as parallel:
            self.descs = np.array(parallel(delayed(self.cdf)(i) for i in _getimages(self.datapath))).squeeze()
        pass
