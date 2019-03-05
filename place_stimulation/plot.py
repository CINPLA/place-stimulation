import matplotlib.pylab as plt
import spatialspikefields as tr
from scipy.interpolate import interp1d
import quantities as pq
import numpy as np
import seaborn as sns


def plot_rate_map(x, y, t, sptr, binsize=0.02, smoothing=0.03, figsize=[5, 5], ax=None):
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

    sptr_t = sptr[sptr.times.magnitude < np.max(t)]
    sptr_t = sptr_t[sptr_t.times.magnitude > np.min(t)]

    x_spike = interp1d(t, x)(sptr_t)
    y_spike = interp1d(t, y)(sptr_t)

    ax.scatter(x_spike, y_spike, s=s, c=c)
    plt.xticks([])
    plt.yticks([])

    return ax


def plot_split_path(sptr, data_path, par, fig=None, figsize=(3, 20), scatter=True):
    if fig is None:
        fig = plt.figure(figsize=figsize)
        # sns.set(color_codes=True, style="darkgrid")

    # making subplots
    nr_channel = 4
    axs = []
    for channel in range(nr_channel):
        ax = plt.subplot2grid((2, 4), (0, channel), rowspan=2, fig=fig)
        axs.append(ax)

    # saving spiketrains from all four tetrodes in channel group
    for channel in range(nr_channel):
        x, y, t, speed = ps.load_tracking(data_path, par, select_tracking=1)    #need to fix loading of the load_tracking()
        sptr_c = sptr[channel][sptr[channel].times.magnitude < np.max(t)]
        sptr_c = sptr_c[sptr_c.times.magnitude > np.min(t)]

        r = np.sqrt(x ** 2 + y ** 2)
        axs[channel].plot(r, t, 'k', alpha=0.3)

        x_spike = interp1d(t, x)(sptr_c)
        y_spike = interp1d(t, y)(sptr_c)
        r_spike = np.sqrt(x_spike ** 2 + y_spike ** 2)
        if scatter:
            axs[channel].scatter(r_spike, t, edgecolor="b")
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xticks([])
            plt.yticks([])

    return fig


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
