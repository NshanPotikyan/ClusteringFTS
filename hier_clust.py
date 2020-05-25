import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from thesis.tools import *
import progressbar

# widgets for the progress bar
widgets = ['Dissimilarity computation: ', progressbar.Percentage(), ' ',
           progressbar.Bar(marker="-", left="[", right="]"),
           ' ', progressbar.ETA()]


class HClust(object):
    """
    Performs hierarchical clustering of given time series
    """

    def __init__(self, data, ground_truth, dist_func,
                 residuals=False, verbose=False, precomputed=False, **kwargs):
        """
        Initializing the clustering object
        :param data: pandas DataFrame of time series data, or dissimilarity matrix, when precomputed=True
        :param ground_truth: numpy array of ground truth cluster labels
        :param dist_func: str name of the dissimilarity function to use, currently supports
               'euclidean', 'dtw', 'corr','corr2', 'cross_corr',
                'cross_corr2','cross_corr3','acf', 'pacf'
        :param residuals: boolean specifying whether data consists of residuals
               from models (used for d_{RCCF} measures)
        :param verbose: boolean controls the verbosity of the algorithm
        :param precomputed: boolean specifying whether data is already a dissimilarity matrix
        :param kwargs:
        """
        self.data = data
        self.ground_truth = ground_truth
        self.nr_clusters = len(np.unique(ground_truth))
        self.progressbar = progressbar.ProgressBar(widgets=widgets)
        self.verbose = verbose
        self.residuals = residuals
        if precomputed:
            self.dist_mat = data
        else:
            self.dist_func = dist_measure(dist_func)
            self.dist_mat = self.compute_distance(**kwargs)
        self.labels = None
        warnings.filterwarnings("ignore")

    def compute_distance(self, **kwargs):
        """
        Computes the dissimilarity matrix between the time series
        :param kwargs:
        :return: pandas DataFrame
        """
        dataset = self.data
        func_name = self.dist_func.__name__
        nr_series = dataset.shape[1]
        out = np.zeros((nr_series, nr_series))

        if ('corr' in func_name) and (not self.residuals):
            # take first order differences of the series in case of correlation-based measures
            dataset = dataset.diff()[1:]

        if ('dtw' in func_name) or ('euclidean' in func_name):
            # standardize the time series in case of DTW and L2 norm
            dataset = pd.DataFrame(normalize(dataset, axis=0))

        if self.verbose:
            iterations = self.progressbar(range(nr_series))
        else:
            iterations = range(nr_series)

        for i in iterations:
            for j in range(i + 1, nr_series):
                out[i, j] = self.dist_func(dataset.iloc[:, i],
                                           dataset.iloc[:, j], **kwargs)

        i_lower = np.tril_indices(len(out), -1)
        out[i_lower] = out.T[i_lower]
        return pd.DataFrame(out)

    def get_clusters(self, nr_clusters=None, linkage_type='single'):
        """
        Apply Agglomerative Clustering on the dissimilarity matrix
        in order to obtain a clustering
        :param nr_clusters: int number of clusters to obtain
        :param linkage_type: str type of linkage to apply (see sklearn documentation for more info)
        :return:
        """
        if nr_clusters is None:
            nr_clusters = self.nr_clusters
        kwargs = dict(n_clusters=nr_clusters,
                      affinity='precomputed',
                      compute_full_tree=True,
                      linkage=linkage_type)
        self.labels = AgglomerativeClustering(**kwargs).fit_predict(self.dist_mat)
        self.labels = relabel(self.labels)

    def cluster_eval(self, nr_clusters=None, eval_type='sim', linkage_type='single'):
        """
        Evaluate a clustering using either Similarity or Silhouette index
        :param nr_clusters: int number of clusters to obtain in the result of clustering
        :param eval_type: str type of the evaluation supports 'sim' or 'sil'
        :param linkage_type: str type of linkage to apply (see sklearn documentation for more info)
        :return:
        """
        if eval_type == 'sim':
            self.get_clusters(nr_clusters=nr_clusters, linkage_type=linkage_type)
            return cluster_sim(ground_truth=self.ground_truth,
                               predicted=self.labels)
        elif eval_type == 'sil':
            self.get_clusters(nr_clusters=nr_clusters, linkage_type=linkage_type)
            return silhouette(self.dist_mat, self.labels)

    def plot_heatmap(self, xlab='Heatmap', reversed_color=True, **kwargs):
        """
        Plots the heatmap of the dissimilarity matrix
        :param xlab: str for x label
        :param reversed_color: boolean for reverting the color bar
        :param kwargs:
        :return:
        """
        if 'figsize' in kwargs:
            plt.figure(figsize=kwargs['figsize'])

        if reversed_color:
            cmap = sns.cm.rocket_r
        else:
            cmap = sns.cm.rocket
        sns.heatmap(self.dist_mat, linewidth=0.5, cmap=cmap)
        nr_series = np.arange(self.data.shape[1])
        plt.yticks(nr_series + 0.5, nr_series + 1, fontsize=12)
        plt.xticks(nr_series + 0.5, nr_series + 1, fontsize=12)
        plt.xlabel(xlab, fontsize=18)

    def plot_dendrogram(self, labels=None, color_threshold=0.8, linkage_type='single'):
        """
        Plot the dendrogram of the clustering
        :param labels: list of str for leaf names
        :param color_threshold: float for coloring the tree linkages as clusters, below the specified value
        :param linkage_type: str type of linkage to apply (see sklearn documentation for more info)
        :return:
        """
        if labels is None:
            labels = [f'$x^{{({i})}}$' for i in np.arange(1, self.data.shape[1]+1)]
        dists = squareform(self.dist_mat.values)
        linkage_matrix = linkage(dists, linkage_type)
        dendrogram(linkage_matrix, color_threshold=color_threshold, labels=labels, show_contracted=True)
        plt.ylabel('Dissimilarity')
