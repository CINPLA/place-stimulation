import matplotlib.pylab as plt
import spatial_maps as sm
from scipy.interpolate import interp1d
import quantities as pq
import numpy as np
import seaborn as sns
from .tools import load_tracking


# Source: Mikkel
# TODO add channel idx
def plot_waveforms(sptr, color='b', fig=None, title=None, lw=2, gs=None, sample_rate=None):
    """
    Visualize waveforms on respective channels
    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure
    Returns
    -------
    out : fig
    """
    import matplotlib.gridspec as gridspec
    nrc = sptr.waveforms.shape[1]
    if fig is None:
        fig = plt.figure(figsize=(20, 3))
        sns.set(color_codes=True, style="darkgrid")
    axs = []
    ax = None
    if sample_rate is None:
        sample_rate = 30 * pq.kHz
    for c in range(nrc):
        if gs is None:
            ax = fig.add_subplot(1, nrc, c+1, sharex=ax, sharey=ax)
        else:
            gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
            ax = fig.add_subplot(gs0[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32)/sample_rate
        stime = stime.rescale('ms')
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlabel(stime.rescale('ms').dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [$\mu$V]')
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_rate_map(x, y, t, sptr, boxsize=1, binsize=0.02, smoothing=0.02, ax=None, mask_zero_occupancy=True):
    map = sm.SpatialMap(x, y, t, sptr, boxsize, binsize)
    rate_map = map.rate_map(smoothing=smoothing, mask_zero_occupancy=mask_zero_occupancy)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(rate_map.T, origin="lower")
    plt.xticks([])
    plt.yticks([])

    return ax


def plot_path(x, y, t, sptr=None, figsize=[5, 5], ax=None, s=30, c1=[0.7, 0.2, 0.2], c2='k', scatter=True):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.plot(x, y, 'k', alpha=0.3)

    if sptr is not None:
        if len(sptr) == 2:
            sptr_t1 = sptr[0][sptr[0].times.magnitude < np.max(t)]
            sptr_t1 = sptr_t1[sptr_t1.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x)(sptr_t1)
            y_spike = interp1d(t, y)(sptr_t1)

            ax.scatter(x_spike, y_spike, s=s, c=c1)

            sptr_t2 = sptr[1][sptr[1].times.magnitude < np.max(t)]
            sptr_t2 = sptr_t2[sptr_t2.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x)(sptr_t2)
            y_spike = interp1d(t, y)(sptr_t2)

            ax.scatter(x_spike, y_spike, s=s, c=c2)
        else:
            sptr_t = sptr[sptr.times.magnitude < np.max(t)]
            sptr_t = sptr_t[sptr_t.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x)(sptr_t)
            y_spike = interp1d(t, y)(sptr_t)

            ax.scatter(x_spike, y_spike, s=s, c=c1)
    ax.axis('equal')
    plt.xticks([])
    plt.yticks([])

    return ax


def plot_split_path(x, y, t, sptr=None, ax=None, figsize=(3, 20),  s=30, c1=[0.7, 0.2, 0.2], c2='k'):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        # sns.set(color_codes=True, style="darkgrid")

    x_center = ((x - (np.max(x) - np.min(x)) / 2) / (np.max(x) - np.min(x))).magnitude
    y_center = ((y - (np.max(y) - np.min(y)) / 2) / (np.max(y) - np.min(y))).magnitude
    angle_position = np.rad2deg(np.arctan2(y_center, x_center))
    ax.plot(angle_position, t, 'k', alpha=0.3)

    if sptr is not None:
        if len(sptr) == 2:
            sptr_t1 = sptr[0][sptr[0].times.magnitude < np.max(t)]
            sptr_t1 = sptr_t1[sptr_t1.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x_center)(sptr_t1)
            y_spike = interp1d(t, y_center)(sptr_t1)

            angle_spikes = np.rad2deg(np.arctan2(y_spike, x_spike))
            ax.scatter(angle_spikes, sptr_t1, s=s, c=c1)

            sptr_t2 = sptr[1][sptr[1].times.magnitude < np.max(t)]
            sptr_t2 = sptr_t2[sptr_t2.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x_center)(sptr_t2)
            y_spike = interp1d(t, y_center)(sptr_t2)

            angle_spikes = np.rad2deg(np.arctan2(y_spike, x_spike))
            ax.scatter(angle_spikes, sptr_t2, s=s, c=c2)
        else:
            sptr_t = sptr[sptr.times.magnitude < np.max(t)]
            sptr_t = sptr_t[sptr_t.times.magnitude > np.min(t)]

            x_spike = interp1d(t, x_center)(sptr_t)
            y_spike = interp1d(t, y_center)(sptr_t)
            angle_spikes = np.rad2deg(np.arctan2(y_spike, x_spike))
            ax.scatter(angle_spikes, sptr_t, s=s, c=c1)

    ax.set_xticks([-180, -90, 0, 90, 180])

    return ax, angle_position


def plot_psth(st, epoch, lags=(-0.1 * pq.s, 10 * pq.s), bin_size=0.01 * pq.s, ax=None, color='C0',
              figsize=[5, 5], n_trials=None):
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
    if n_trials is None:
        n_trials = len(epoch)

    sts = []
    for h, epo in enumerate(epoch):
        if h < n_trials:
            st_ = st.time_slice(t_start=epo + lags[0],
                                t_stop=epo + lags[1])
            sts.append((st_.times - epo).simplified.tolist())
            ax.plot(sts[h], np.zeros(len(sts[h])) + h+1, '|', color=color)

    ax.set_title('PSTH')
    ax.axvline(0)
    ax.set_xlabel('lag (s)')
    flatten_sts = [item for sublist in sts for item in sublist]
    if len(flatten_sts) > 0:
        flatten_sts = flatten_sts / np.max(flatten_sts) * (n_trials-1)
        ax.hist(flatten_sts, bins=bins, color=color, alpha=0.3)
        ax.set_xticks([lags[0], 0*pq.s, lags[1]])
