import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm
from sklearn.metrics import silhouette_score


def get_synthetic_data(path, classes=(0, 1, 0, 1)):
    """
    Imports the synthetic data from csv file
    :param path: the path of the file
    :param classes: the classes of the time series
    :return: pandas DataFrame for the synthetic data,
             numpy array of the ground truth labels
    """
    df = pd.read_csv(path)
    nr_series = df.shape[1]
    nr_classes = len(classes)
    nr_series_per_class = nr_series // nr_classes
    df.columns = [f'x{i+1}' for i in np.arange(nr_series)]
    true_cluster = np.concatenate([nr_series_per_class * [i] for i in classes])
    return df, true_cluster


def plot_shift(ts, lag=0, **kwargs):
    """
    Plots a single time series and colors the parts that has been shifted
    :param ts: numpy array of time series
    :param lag: int the amount of shift
    :param kwargs:
    :return:
    """
    plt.plot(np.arange(lag+1), ts[:(lag+1)], c='C1', linewidth=3, **kwargs)
    plt.plot(np.arange(lag, len(ts)), ts[lag:], c='C0', linewidth=3, **kwargs)
    plt.xlabel('Time', size=15)


def plot_multiple(data, lags, ticks, add_shift, tick_func=np.mean):
    """
    Plots multiple series of the same class (in the synthetic dataset)
    :param data: pandas DataFrame, where each column is a time series
    :param lags: int to specify the time delay of the series w.r.t the first series
    :param ticks: list or numpy array for the tick numbers (x^ticks)
    :param add_shift: float specifying the amount of shift for visual purposes
    :param tick_func: aggregating function for placing the yticks
    :return:
    """
    assert data.shape[1] == len(lags)
    initial_values = []  # for yticks
    shift = 0
    for i in range(data.shape[1]):
        ts = ts_normalized(data.iloc[:, i]) + shift
        plot_shift(ts, lags[i])
        initial_values.append(tick_func(ts))
        shift += add_shift
    plt.yticks(initial_values, [f'$x^{{({i})}}$' for i in ticks], size=15)


def plot_series(df, nr_series_per_class, add_shift=1, save=False):
    """
    Plots the synthetic time series of each class
    :param df: pandas DataFrame, where each column is a time series
    :param nr_series_per_class:
    :param add_shift: shift is added for visual purposes
    :param save: boolean specifying whether to save the figure or not
    :return:
    """
    lags = np.arange(nr_series_per_class)
    nr_classes = df.shape[1] // nr_series_per_class

    for i in range(nr_classes):
        plot_multiple(df.iloc[:, np.arange(i*nr_series_per_class, (i+1)*nr_series_per_class)],
                      lags=lags, ticks=lags+i*nr_series_per_class+1,
                      add_shift=add_shift, tick_func=np.mean)
        if save:
            plt.savefig('pdf/class{}.png'.format(i+1))
        plt.show()
   
   
def plot_one_per_class(df, nr_series_per_class=5):
    """
    Plots one synthetic time series from each class
    :param df: pandas DataFrame, where each column is a time series
    :param nr_series_per_class:
    :return:
    """
    nr_classes = df.shape[1] // nr_series_per_class
    for i in range(nr_classes):
        plt.plot(df.iloc[:, i * nr_series_per_class],
                 label=f'$x^{{({i*nr_series_per_class + 1})}} \in C_{i+1}$',
                 linewidth=3)
    plt.legend(fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.show()


def get_similarities(diss_mat, ground_truth):
    """
    Given the dissimilarity matrix and the ground-truth labels
    computes the similarity index for clusterings obtained with 
    single, complete and average linkage methods 
    :diss_mat: pandas DataFrame of the dissimilarity matrix
    :return: pandas DataFrame with one row
    """
    nr_clusters = len(np.unique(ground_truth))
    clustering1 = HClust(data=diss_mat, ground_truth=ground_truth,
                         dist_func=None, precomputed=True)

    out = {'single': [],
           'complete': [],
           'average': []}

    for linkage in out: 
        out[linkage].append(clustering1.cluster_eval(nr_clusters=nr_clusters,
                                                     linkage_type=linkage,
                                                     eval_type='sim'))
    return pd.DataFrame(out)


def dist_measure(dist):
    """
    Gives the function from its name
    :param dist: str name of the dissimilarity function
    :return: dissimilarity function
    """
    measures = {'euclidean': euclidean,
                'dtw': dtw,
                'corr1': corr,
                'corr2': corr2,
                'cross_corr1': cross_corr,
                'cross_corr2': cross_corr2,
                'cross_corr3': cross_corr3,
                'acf': acf_diss,
                'pacf': pacf_diss
                }

    assert dist in measures
    return measures[dist]


def ts_normalized(input_series):
    """
    MinMax scaling of the given time series
    :param input_series: numpy array or pandas Series
    :return: numpy array or pandas Series
    """
    return (input_series - input_series.min()) / (input_series.max() - input_series.min())


def minkowski(s1, s2, p=2):
    """
    Computes the minkowski distance between given time serise
    :param s1: time series 1
    :param s2: time series 2
    :param p: positive integer
    :return: dissimilarity measure
    """

    return (abs(s1 - s2) ** p).sum() ** (1/p)


def euclidean(s1, s2, **kwargs):
    """
    d_{L_2} distance measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    return minkowski(s1, s2)


def dtw(s1, s2, **kwargs):
    """
    d_{DTW} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    return fastdtw(s1, s2)[0]


def corr(s1, s2, **kwargs):
    """
    d_{COR_1} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    cor = np.corrcoef(s1, s2)[0, 1]
    return np.sqrt(2 * (1 - cor))


def corr2(s1, s2, **kwargs):
    """
    d_{COR_2} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    if 'beta' not in kwargs:
        beta = 1
    else:
        beta = kwargs['beta']
    cor = np.corrcoef(s1, s2)[0, 1]
    return np.sqrt(((1 - cor) / (1 + cor)) ** beta)


def ccf(s1, s2, max_lag=10):
    """
    Computes the sample cross-correlation between given series
    :param s1: time series 1
    :param s2: time series 2
    :param max_lag:
    :return: numpy array of cross-correlation values
    """
    s1 -= s1.mean()
    s2 -= s2.mean()

    out = np.array([np.sum(s1.shift(i) * s2) for i in range(-max_lag, max_lag+1)])
    out /= np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    return out


def cross_corr(s1, s2, **kwargs):
    """
    d_{CCF_1} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    if 'max_lag' not in kwargs:
        max_lag = 10
    else:
        max_lag = kwargs['max_lag']
    out = ccf(s1, s2, max_lag)
    return np.sqrt((1-out[10]**2) / np.sum(out[11:]**2))


def cross_corr2(s1, s2, **kwargs):
    """
    d_{CCF_2} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """

    if 'max_lag' not in kwargs:
        max_lag = 10
    else:
        max_lag = kwargs['max_lag']
    out = ccf(s1, s2, max_lag)
    return np.sqrt(2 * (1 - np.max(out)))


def cross_corr3(s1, s2, **kwargs):
    """
    d_{CCF_3} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    if 'max_lag' not in kwargs:
        max_lag = 10
    else:
        max_lag = kwargs['max_lag']
    lags = np.arange(-max_lag, max_lag+1)
    out = ccf(s1, s2, max_lag)
    weights = norm.pdf(lags, scale=10) / sum(norm.pdf(lags, scale=10))
    out *= weights
    idx = np.argmax(abs(out))
    return np.sqrt(2 * (1 - out[idx] / weights[idx]))


def acf_diss(s1, s2, **kwargs):
    """
    d_{ACF} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    nlag = len(s1) - 1
    return minkowski(acf(s1, nlags=nlag),
                     acf(s2, nlags=nlag))


def pacf_diss(s1, s2, **kwargs):
    """
    d_{PACF} dissimilarity measure
    :param s1: time series 1
    :param s2: time series 2
    :param kwargs:
    :return: dissimilarity measure
    """
    nlag = len(s1) - 1
    return minkowski(pacf(s1, nlags=nlag),
                     pacf(s2, nlags=nlag))


def cluster_sim(ground_truth, predicted):
    """
    Computes the similarity index between ground truth and obtained clusters
    :param ground_truth: numpy array of indices
    :param predicted: numpy array of indices
    :return: similarity score
    """
    def sim(g, p): return 2*len(g & p) / (len(g) + len(p))

    uniques = np.unique(ground_truth)
    sims = np.zeros((len(uniques), len(uniques)))
    for i_idx, i in enumerate(uniques):
        for j_idx, j in enumerate(uniques):
            gi = set(np.where(ground_truth == i)[0])
            pj = set(np.where(predicted == j)[0])
            sims[i_idx, j_idx] = sim(gi, pj)

    return sims.max(axis=0).sum() / len(uniques)


def silhouette(dist_mat, labels):
    """
    Computes the silhouette index for precomputed distance/dissimilarity matrix
    :param dist_mat: numpy array of a squared, symmetric matrix of dissimilarities
    :param labels: numpy array of clustering labels
    :return: float silhouette score
    """
    return silhouette_score(dist_mat, labels, metric='precomputed')


def index_match(a, b):
    """
    Helper function to get the indices of elements in one list
    that appear in another list
    :param a: list or numpy array of elements
    :param b: list or numpy array of elements
    :return: numpy array of indices
    """
    return np.array([j for i in range(len(a)) for j in range(len(a)) if a[i] == b[j]])

def relabel(old_labels):
    """
    Used for sorting the clustering labels in ascending order
    :param old_labels: numpy array of cluster labels
    :return: numpy array of modified labels
    """
    uniques, idx = np.unique(old_labels, return_index=True)
    vals = {k: v for v, k in zip(uniques, old_labels[np.sort(idx)])}
    return np.array([vals[old_labels[i]] for i in range(len(old_labels))])



