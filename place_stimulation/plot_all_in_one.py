import matplotlib.pylab as plt
import spatialspikefields as tr
from scipy.interpolate import interp1d
import quantities as pq
import numpy as np
import seaborn as sns
from .tools import load_tracking


def plot_all_recordings(sptr,  data_path, color='b', lw=2, par=None, s=30, c=[0.7, 0.2, 0.2],
                        select_tracking=None, fig=None, figsize=(20, 20),
                        binsize = 0.02, smoothing = 0.03):
    gs = gridspec.GridSpec(len(sptr), 4)
    if len(sptr) == 0:
        return "No cells in this channel group"

    # creating axes for waveforms
    gs_wf = gridspec.GridSpec(len(sptr), 2, subplot_spec=gs)
    gs_wf.update(left=0.05, right=0.48)
    axs_wf = []
    axs_wf.append(plt.subplot(gs_wf[0, :], sharex=True))
    for i in range(1, len(sptr)):
        axs_wf.append(plt.subplot(gs_wf[1, :]))

    for idx in range(len(sptr)):
        wf = sptr[i].waveforms[:, i, :]
        m = np.mean(wf, axis=0)
        s_time = np.arange(m.size, dtype=np.float32) / sptr.sampling_rate
        s_time.units = 'ms'
        sd = np.std(wf, axis=0)
        axs_wf[i].plot(s_time, m, color=color, lw=lw)
        axs_wf[i].fill_between(s_time, m - sd, m + sd, alpha=.1, color=color)
        #if sptr.left_sweep is not None:
         #   sptr.left_sweep.units = 'ms'
          #  axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
           #                ls='--')
        axs_wf[i].set_xlabel(s_time.dimensionality)
        axs_wf[i].set_xlim([s_time.min(), s_time.max()])
        if idx != 0:
            plt.setp(axs_wf[idx].get_yticklabels(), visible=False)

    # plotting path
    if par is None:
        par = {'speed_filter': 0.3,  # m/s
               'pos_fs': 60,
               'f_cut': 1,
               'spat_binsize': 0.02,
               'spat_smoothing': 0.025,
               'grid_stepsize': 0.1,
               'box_xlen': 1,
               'box_ylen': 1,
               'spike_size': 10,
               'field_max_wall_extent': 0.1,
               'field_min_bins': 12
               }

    # creating axes for path
    x, y, t, speed = load_tracking(data_path, par, select_tracking=1)
    gs_p = gridspec.GridSpec(len(sptr), 1, subplot_spec=gs)
    gs_p.update(left=0.55, right=0.98)
    axs_p = [plt.subplot(gs_p[0, :], sharex=True)]
    for i in range(1, len(sptr)):
        axs_p.append(plt.subplot(gs_p[i, :]))
        axs_p[i].plot(x, y, 'k', alpha=0.3)

    x_spikes = np.zeros(4)
    y_spikes = np.zeros(4)
    for i in range(4):
        sptr_t = sptr[i][sptr[i].times.magnitude < np.max(t)]
        sptr_t = sptr_t[sptr_t.times.magnitude > np.min(t)]
        x_spikes[i] = interp1d(t, x)(sptr_t)
        y_spikes[i] = interp1d(t, y)(sptr_t)

    for ax, i in zip(axs_p, range(len(sptr))):
        ax.scatter(x_spikes[i], y_spikes[i], s=s, c=c)

    plt.xticks([])
    plt.yticks([])

    # plotting firing-rate maps
    gs_fr = gridspec.GridSpec(len(sptr), 1)
    gs_fr.update(left=1.05, right=1.48)
    axs_fr = [plt.subplot(gs_p[0, :], sharex=True)]
    rate_map = tr.spatial_rate_map(x, y, t, sptr[0], binsize=binsize, box_xlen=1, box_ylen=1, smoothing=smoothing)
    axs_fr[0].imshow(rate_map, origin="lower")
    for i in range(1, len(sptr)):
        axs_fr.append(plt.subplot(gs_p[i, :]))
        rate_map = tr.spatial_rate_map(x, y, t, sptr[i], binsize=binsize, box_xlen=1, box_ylen=1, smoothing=smoothing)
        axs_fr[i].imshow(rate_map, origin="lower")

    plt.xticks([])
    plt.yticks([])

    plt.savefig("all-in-one-plot.png")

    return plt.show()