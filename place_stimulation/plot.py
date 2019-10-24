import matplotlib.pylab as plt
import spatial_maps as sm
from scipy.interpolate import interp1d
import quantities as pq
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import neo
from .tools import load_tracking, load_spiketrains, load_epochs, get_data_path, find_putative_target_cell, \
    remove_central_region, compute_rate_map


# Source: Mikkel
def plot_waveforms(sptr, color='b', title=None, lw=2, ax=None, sample_rate=None, ylim=None):
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
    nrc = sptr.waveforms.shape[1]
    if ax is None:
        fig = plt.figure(figsize=(20, 3))
        ax = fig.add_subplot(111)
        sns.set(color_codes=True, style="darkgrid")
    else:
        fig = ax.get_figure()
    axs = []
    gs = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=ax)

    if sample_rate is None:
        sample_rate = sptr.sampling_rate
    for c in range(nrc):
        ax = fig.add_subplot(gs[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32) / sample_rate
        stime = stime.rescale('ms')
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m - sd, m + sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlim([stime.min(), stime.max()])
        if 'channel_idx' in sptr.annotations.keys():
            if len(sptr.annotations['channel_idx']) == nrc:
                axs[c].set_title(str(sptr.annotations['channel_idx'][c]))
        axs[c].set_xticklabels([])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
        if ylim is not None:
            axs[c].set_ylim(ylim)
        # axs[c].axis('off')
    # axs[0].set_ylabel('amplitude (uV)')
    # ax.axis('off')
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_rate_map(x, y, t, sptr, box_size=1, bin_size=0.02, smoothing=0.02, ax=None, mask_zero_occupancy=True):
    map = sm.SpatialMap(x, y, t, sptr, box_size, bin_size)
    rate_map = map.rate_map(smoothing=smoothing, mask_zero_occupancy=mask_zero_occupancy)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(rate_map.T, origin="lower")
    plt.xticks([])
    plt.yticks([])

    return ax


def plot_smoothed_rate_map(x, y, t, sptr, box_size, bin_size, xregion, yregion, smoothing, ax):
    rate_map = compute_rate_map(x, y, t, sptr, box_size, bin_size, xregion, yregion, smoothing)

    ax.imshow(rate_map.T, origin='lower')


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


def plot_split_path(x, y, t, sptr=None, ax=None, figsize=(3, 20), s=30, c1=[0.7, 0.2, 0.2], c2='k'):
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
            ax.plot(sts[h], np.zeros(len(sts[h])) + h + 1, '|', color=color)

    ax.set_title('PSTH')
    ax.axvline(0)
    ax.set_xlabel('lag (s)')
    flatten_sts = [item for sublist in sts for item in sublist]
    if len(flatten_sts) > 0:
        flatten_sts = flatten_sts / np.max(flatten_sts) * (n_trials - 1)
        ax.hist(flatten_sts, bins=bins, color=color, alpha=0.3)
        ax.set_xticks([lags[0], 0 * pq.s, lags[1]])


def session_summary(cell_df, actions, **kwargs_spatial):
    if len(cell_df) > 0:
        fig = plt.figure(figsize=(15, 10))
        ncols = len(cell_df)
        if ncols == 1:
            ncols = 2
        nrows = 4

        interp = kwargs_spatial['interp']
        quantiles = kwargs_spatial['quantiles']
        fc = kwargs_spatial['fc']
        xregion = kwargs_spatial['xregion']
        yregion = kwargs_spatial['yregion']
        smoothing_low = kwargs_spatial['smoothing_low']
        smoothing_high = kwargs_spatial['smoothing_high']
        bin_size = kwargs_spatial['bin_size']

        for i, (unit_index, unit_row) in enumerate(cell_df.iterrows()):
            action = unit_row['action']
            unit_id = unit_row['unit_id']
            matched_id = unit_row['unique_unit_id']
            ch_group = unit_row['channel_group']

            data_path = get_data_path(actions[action])
            tags = actions[action].tags

            if 'intan' in tags:
                #                 print('Intan')
                epochs = load_epochs(data_path)
                if len(epochs) == 4:
                    # stimulation trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[2].times[0])
                elif len(epochs) == 2:
                    # non-stim trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[1].times[0])
                else:
                    print('Wrong epochs')
            else:
                #                 print('OE')
                x, y, t, speed = load_tracking(data_path, select_tracking=0, interp=interp, fc=fc,
                                               remove_quantiles=quantiles)
                sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id, load_waveforms=True)
            box_size = np.array([np.max(x), np.max(y)])
            x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

            ax_rm = fig.add_subplot(nrows, ncols, i + 1)
            plot_smoothed_rate_map(x, y, t, sptr[0], box_size=box_size, bin_size=bin_size, xregion=xregion,
                                   yregion=yregion,
                                   smoothing=smoothing_low, ax=ax_rm)

            ax_pp = fig.add_subplot(nrows, ncols, ncols + i + 1)
            plot_path(x, y, t, sptr[0], ax=ax_pp)
            ax_pp.axis('off')
            ylim = [-300, 100]
            ax_wf = fig.add_subplot(nrows, ncols, 2 * ncols + i + 1)
            ax_wf.axis('off')
            plot_waveforms(sptr[0], ax=ax_wf, ylim=ylim)
            ax_pp.axis('off')
            ax_rm.set_title(str(ch_group) + '-' + str(unit_id) + '-' + str(matched_id), fontsize=8)

        stim_action = cell_df.iloc[0].stim_action
        data_path = get_data_path(actions[stim_action])
        tags = actions[stim_action].tags
        if 'intan' in tags:
            # print('Intan recording')
            epochs = load_epochs(data_path)
            if len(epochs) == 4:
                # stimulation trial
                times = epochs[1].times - epochs[0].times[0]
                sptr = [neo.SpikeTrain(times=times, t_start=0 * pq.s, t_stop=times[-1])]
                x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                               remove_quantiles=quantiles, t_start=epochs[2].times[0])
                box_size = np.array([np.max(x), np.max(y)])
                x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

                ax_rm = fig.add_subplot(nrows, ncols, 3 * ncols + 1)
                plot_smoothed_rate_map(x, y, t, sptr, box_size=box_size, bin_size=bin_size, xregion=xregion,
                                       yregion=yregion,
                                       smoothing=smoothing_high, ax=ax_rm)
                ax_pp = fig.add_subplot(nrows, ncols, 3 * ncols + 2)
                plot_path(x, y, t, sptr[0], ax=ax_pp, c1=[0.3, 0.3, 0.3])
                ax_pp.axis('off')
                ax_rm.set_title('stim rate map')
                ax_pp.set_title('stim rate pulses')
        return fig
    else:
        print('No target cells in this session:', np.unique(cell_df['stimulation']))
        return None


def plot_target_cell(pre, stim, post, actions, target_id=None, smoothing_rm='low', smoothing_st='high',
                     **kwargs_spatial):
    if target_id is None:
        target_id, match = find_putative_target_cell(pre, stim, post)
    else:
        unique_id_pre = pre.unique_unit_id
        unique_id_stim = stim.unique_unit_id
        unique_id_post = post.unique_unit_id

        unique_id_pre = unique_id_pre[unique_id_pre.isnull() == False].to_list()
        unique_id_stim = unique_id_stim[unique_id_stim.isnull() == False].to_list()
        unique_id_post = unique_id_post[unique_id_post.isnull() == False].to_list()

        if target_id not in unique_id_pre:
            raise Exception('target cell not in pre-stim list')

        match = None

        if target_id in unique_id_post and target_id in unique_id_stim:
            match = 'all'
        elif target_id in unique_id_post:
            match = 'post'
        elif target_id in unique_id_stim:
            match = 'stim'

    interp = kwargs_spatial['interp']
    quantiles = kwargs_spatial['quantiles']
    fc = kwargs_spatial['fc']
    xregion = kwargs_spatial['xregion']
    yregion = kwargs_spatial['yregion']
    smoothing_low = kwargs_spatial['smoothing_low']
    smoothing_high = kwargs_spatial['smoothing_high']
    bin_size = kwargs_spatial['bin_size']

    if smoothing_rm == 'high':
        smoothing_rate = smoothing_high
    else:
        smoothing_rate = smoothing_low
    if smoothing_st == 'high':
        smoothing_stim = smoothing_high
    else:
        smoothing_stim = smoothing_low

    if target_id is not None:
        fig = plt.figure(figsize=(15, 10))
        ncols = 3
        nrows = 4

        # plot pre
        unit_row = pre[pre.unique_unit_id == target_id].iloc[0]
        action = unit_row['action']
        unit_id = unit_row['unit_id']
        matched_id = unit_row['unique_unit_id']
        ch_group = unit_row['channel_group']

        data_path = get_data_path(actions[action])
        tags = actions[action].tags

        if 'intan' in tags:
            epochs = load_epochs(data_path)
            if len(epochs) == 4:
                # stimulation trial
                sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                        t_start=epochs[0].times[0], load_waveforms=True)
                x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                               remove_quantiles=quantiles, t_start=epochs[2].times[0])
            elif len(epochs) == 2:
                # non-stim trial
                sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                        t_start=epochs[0].times[0], load_waveforms=True)
                x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                               remove_quantiles=quantiles, t_start=epochs[1].times[0])
            else:
                print('Wrong epochs')
        else:
            x, y, t, speed = load_tracking(data_path, select_tracking=0, interp=interp, fc=fc,
                                           remove_quantiles=quantiles)
            sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id, load_waveforms=True)
        box_size = np.array([np.max(x), np.max(y)])
        x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

        ax_rm = fig.add_subplot(nrows, ncols, 1)
        plot_smoothed_rate_map(x, y, t, sptr[0], box_size=box_size, bin_size=bin_size, smoothing=smoothing_rate,
                               xregion=xregion, yregion=yregion, ax=ax_rm)
        ax_pp = fig.add_subplot(nrows, ncols, 4)
        plot_path(x, y, t, sptr[0], ax=ax_pp)
        ax_pp.axis('off')
        ylim = [-300, 100]
        ax_wf = fig.add_subplot(nrows, ncols, 7)
        plot_waveforms(sptr[0], ax=ax_wf, ylim=ylim)
        ax_wf.axis('off')
        ax_pp.axis('off')
        ax_rm.set_title('PRE', fontsize=15)

        # plot stim
        if match in ['all', 'stim']:
            unit_row = stim[stim.unique_unit_id == target_id].iloc[0]
            action = unit_row['action']
            unit_id = unit_row['unit_id']
            matched_id = unit_row['unique_unit_id']
            ch_group = unit_row['channel_group']

            data_path = get_data_path(actions[action])
            tags = actions[action].tags

            if 'intan' in tags:
                #                 print('Intan')
                epochs = load_epochs(data_path)
                if len(epochs) == 4:
                    # stimulation trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[2].times[0])
                elif len(epochs) == 2:
                    # non-stim trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[1].times[0])
                else:
                    print('Wrong epochs')
            else:
                #                 print('OE')
                x, y, t, speed = load_tracking(data_path, select_tracking=0, interp=interp, fc=fc,
                                               remove_quantiles=quantiles)
                sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id, load_waveforms=True)
            box_size = np.array([np.max(x), np.max(y)])
            x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

            ax_rm = fig.add_subplot(nrows, ncols, 2)
            plot_smoothed_rate_map(x, y, t, sptr[0], box_size=box_size, bin_size=bin_size, smoothing=smoothing_rate,
                                   xregion=xregion, yregion=yregion, ax=ax_rm)
            ax_pp = fig.add_subplot(nrows, ncols, 5)
            plot_path(x, y, t, sptr[0], ax=ax_pp)
            ax_pp.axis('off')
            ylim = [-300, 100]
            ax_wf = fig.add_subplot(nrows, ncols, 8)
            plot_waveforms(sptr[0], ax=ax_wf, ylim=ylim)
            ax_wf.axis('off')
            ax_pp.axis('off')
            ax_rm.set_title('STIM', fontsize=15)

        # plot ost
        if match in ['all', 'post']:
            unit_row = post[post.unique_unit_id == target_id].iloc[0]
            action = unit_row['action']
            unit_id = unit_row['unit_id']
            matched_id = unit_row['unique_unit_id']
            ch_group = unit_row['channel_group']

            data_path = get_data_path(actions[action])
            tags = actions[action].tags

            if 'intan' in tags:
                #                 print('Intan')
                epochs = load_epochs(data_path)
                if len(epochs) == 4:
                    # stimulation trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[2].times[0])
                elif len(epochs) == 2:
                    # non-stim trial
                    sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id,
                                            t_start=epochs[0].times[0], load_waveforms=True)
                    x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                                   remove_quantiles=quantiles, t_start=epochs[1].times[0])
                else:
                    print('Wrong epochs')
            else:
                #                 print('OE')
                x, y, t, speed = load_tracking(data_path, select_tracking=0, interp=interp, fc=fc,
                                               remove_quantiles=quantiles)
                sptr = load_spiketrains(data_path, channel_group=ch_group, unit_id=unit_id, load_waveforms=True)
            box_size = np.array([np.max(x), np.max(y)])
            x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

            ax_rm = fig.add_subplot(nrows, ncols, 3)
            plot_smoothed_rate_map(x, y, t, sptr[0], box_size=box_size, bin_size=bin_size, smoothing=smoothing_rate,
                                   xregion=xregion, yregion=yregion, ax=ax_rm)
            ax_pp = fig.add_subplot(nrows, ncols, 6)
            plot_path(x, y, t, sptr[0], ax=ax_pp)
            ax_pp.axis('off')
            ylim = [-300, 100]
            ax_wf = fig.add_subplot(nrows, ncols, 9)
            plot_waveforms(sptr[0], ax=ax_wf, ylim=ylim)
            ax_wf.axis('off')
            ax_pp.axis('off')
            ax_rm.set_title('POST', fontsize=15)

        stim_action = pre.iloc[0].stim_action
        data_path = get_data_path(actions[stim_action])
        tags = actions[stim_action].tags

        if 'intan' in tags:
            # print('Intan recording')
            epochs = load_epochs(data_path)
            if len(epochs) == 4:
                # stimulation trial
                times = epochs[1].times - epochs[0].times[0]
                sptr = [neo.SpikeTrain(times=times, t_start=0 * pq.s, t_stop=times[-1])]
                x, y, t, speed = load_tracking(data_path, interp=interp, select_tracking=0, fc=fc,
                                               remove_quantiles=quantiles, t_start=epochs[2].times[0])
                box_size = np.array([np.max(x), np.max(y)])
                x, y, t = remove_central_region(x, y, t, xregion * np.max(x), yregion * np.max(y))

                ax_rm = fig.add_subplot(nrows, ncols, 10)
                plot_smoothed_rate_map(x, y, t, sptr, box_size=box_size, bin_size=bin_size, smoothing=smoothing_stim,
                                       xregion=xregion, yregion=yregion, ax=ax_rm)
                ax_pp = fig.add_subplot(nrows, ncols, 11)
                plot_path(x, y, t, sptr[0], ax=ax_pp, c1=[0.3, 0.3, 0.3])
                ax_pp.axis('off')
                ax_rm.set_title('stim rate map')
                ax_pp.set_title('stim rate pulses')
        return target_id, match, fig
    else:
        print('No target cell found')
        return None, None, None