import matplotlib.pylab as plt
import spatialspikefields as tr
from scipy.interpolate import interp1d
import quantities as pq
import numpy as np


def plot_rate_map(x, y, t, sptr, binsize=0.02, smoothing=0.025, figsize=[5, 5], ax=None):
    rate_map = tr.spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                   box_xlen=1, box_ylen=1, smoothing=smoothing)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.imshow(rate_map, origin="lower")
    plt.xticks([])
    plt.yticks([])

    return ax


def plot_path(x, y, t, sptr, figsize=[5, 5], ax=None, s=30, c=[0.7, 0.2, 0.2], scatter=True):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.plot(x, y, 'k', alpha=0.3)

    x_spike = interp1d(t, x)
    y_spike = interp1d(t, y)

    if scatter:
        ax.scatter(x_spike(t), y_spike(t), s=s, c=c, edgecolor="b")
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks([])
        plt.yticks([])

    return ax


def plot_psth(st, epoch, lags=(-0.1 * pq.s, 10 * pq.s), bin_size=0.01 * pq.s, ax=None, color='C0',
              figsize=[5, 5], n_trials=10):
    '''
    Parameters:
    st : neo.SpikeTrain
    epoch : neo.Epoch
    lags : tuple of Quantity scalars
    bin_size : Quantity scalar
    color : mpl color
    n_trials : int
        number of trials to include in PSTH
    '''

    labels = np.unique(epoch.labels, axis=-1)
    bins = np.linspace(lags[0], lags[1], int((lags[1] - lags[0]) // bin_size) + 1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # for i, label in enumerate(labels):
    #     print(i, labels)
    #
    #     ax.set_xlim(lags)
    #     ax.set_xlim(lags)
    #
    sts = []
    for h, epo in enumerate(epoch):
        if h < n_trials:
            st_ = st.time_slice(t_start=epo + lags[0],
                                t_stop=epo + lags[1])
            sts.append((st_.times - epo).simplified.tolist())
            ax.plot(sts[h], np.zeros(len(sts[h])) + h, '|', color=color)

    ax.set_title('PSTH')
    # ax.set_xticklabels([])
    ax.axvline(0)
    ax.set_xlabel('lag (s)')
    flatten_sts = [item for sublist in sts for item in sublist]
    ax.hist(flatten_sts, bins=bins, color=color)
