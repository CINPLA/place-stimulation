# This is work in progress,
import neo
import numpy as np
import exdir
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib
import os
import quantities as pq
import spikeextractors as se


def project_path():
    path = os.environ.get("SEPTUM_MEC_DATA")
    if path is None:
        raise Exception("Need to set `SEPTUM_MEC_DATA` as environment variable first.")
    else:
        path = pathlib.Path(path)
    return path


def view_active_channels(action, sorter):
    path = get_data_path(action)
    elphys_path = path / 'processing' / 'electrophysiology'
    sorter_path = elphys_path / 'spikesorting' / sorter / 'phy'
    return np.load(sorter_path / 'channel_map_si.npy')


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return out


def _read_epoch(exdir_file, path, lazy=False):
    group = exdir_file[path]
    if lazy:
        times = []
    else:
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])

    if "durations" in group and not lazy:
        durations = pq.Quantity(group['durations'].data, group['durations'].attrs['unit'])
    elif "durations" in group and lazy:
        durations = []
    else:
        durations = None

    if 'data' in group and not lazy:
        if 'unit' not in group['data'].attrs:
            labels = group['data'].data
        else:
            labels = pq.Quantity(group['data'].data,
                                 group['data'].attrs['unit'])
    elif 'data' in group and lazy:
        labels = []
    else:
        labels = None
    annotations = {'exdir_path': path}
    annotations.update(group.attrs.to_dict())

    if lazy:
        lazy_shape = (group.attrs['num_samples'],)
    else:
        lazy_shape = None
    epo = neo.Epoch(times=times, durations=durations, labels=labels,
                lazy_shape=lazy_shape, **annotations)

    return epo


def velocity_filter(x, y, t, threshold):
    """
    Removes values above threshold
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    threshold : float
    """
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    vel = np.array([dx, dy]) / dt
    speed = np.linalg.norm(vel, axis=0)
    speed_mask = (speed < threshold)
    speed_mask = np.append(speed_mask, 0)
    x = x[np.where(speed_mask)]
    y = y[np.where(speed_mask)]
    t = t[np.where(speed_mask)]
    return x, y, t


def interp_filt_position(x, y, tm, box_xlen=1 , box_ylen=1 ,
                         fs=100 , f_cut=10 ):
    """
    rapid head movements will contribute to velocity artifacts,
    these can be removed by low-pass filtering
    see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    code addapted from Espen Hagen
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    tm : quantities.Quantity array in s
        1d vector of times at x, y positions
    fs : quantities scalar in Hz
        return radians
    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    t = np.arange(tm.min(), tm.max() + 1. / fs, 1. / fs)
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filteringpar
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(N=1, Wn=f_cut * 2 / fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))
    if (x.min() < 0 or x.max() > box_xlen or y.min() < 0 or y.max() > box_ylen):
        raise ValueError(
            "Interpolation produces path values " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))

    return x, y, t


def rm_nans(*args):
    """
    Removes nan from all corresponding arrays
    Parameters
    ----------
    args : arrays, lists or quantities which should have removed nans in
           all the same indices
    Returns
    -------
    out : args with removed nans
    """
    nan_indices = []
    for arg in args:
        nan_indices.extend(np.where(np.isnan(arg))[0].tolist())
    nan_indices = np.unique(nan_indices)
    out = []
    for arg in args:
        out.append(np.delete(arg, nan_indices))
    return out


def unit_path(channel_id, unit_id):
    return "/processing/electrophysiology/channel_group_{}/UnitTimes/{}".format(channel_id, unit_id)


def load_leds(data_path):
    root_group = exdir.File(
        data_path, "r",
        plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])

    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    stop_time = position_group.attrs["stop_time"]
    x1, y1 = position_group['led_0']['data'].data.T
    t1 = position_group['led_0']['timestamps'].data
    x2, y2 = position_group['led_1']['data'].data.T
    t2 = position_group['led_1']['timestamps'].data

    return x1, y1, t1, x2, y2, t2, stop_time


def filter_xy_zero(x, y, t):
    idxs, = np.where((x == 0) & (y == 0))
    return [np.delete(a, idxs) for a in [x, y, t]]


def load_head_direction(data_path, pos_fs, f_cut):
    from head_direction.head import head_direction
    x1, y1, t1, x2, y2, t2, stop_time = load_leds(data_path)

    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)

    x1, y1, t1 = filter_xy_zero(x1, y1, t1)
    x2, y2, t2 = filter_xy_zero(x2, y2, t2)

    x1, y1, t1 = interp_filt_position(x1, y1, t1, fs=pos_fs, f_cut=f_cut)
    x2, y2, t2 = interp_filt_position(x2, y2, t2, fs=pos_fs, f_cut=f_cut)

    x1, y1, t1, x2, y2, t2 = _cut_to_same_len(x1, y1, t1, x2, y2, t2)

    mask1 = t1 <= stop_time
    mask2 = t2 <= stop_time
    #mask = t2 <= stop_time
    x1, y1, t1 = x1[mask1], y1[mask1], t1[mask1]
    x2, y2, t2 = x2[mask2], y2[mask2], t2[mask2]
    angles, times = head_direction(x1, y1, x2, y2, t1)
    return angles, times


def load_tracking(data_path, sampling_rate, low_pass_frequency, velocity_threshold=5):
    x1, y1, t1, x2, y2, t2, stop_time = load_leds(data_path)
    x1, y1, t1, x2, y2, t2 = [a.magnitude for a in [x1, y1, t1, x2, y2, t2]]
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)

    # select data with least nan
    if len(x1) > len(x2):
        x, y, t = x1, y1, t1
    else:
        x, y, t = x2, y2, t2

    # OE saves 0.0 when signal is lost, these can be removed
    x, y, t = filter_xy_zero(x, y, t)

    # remove velocity artifacts
    x, y, t = velocity_filter(x, y, t, velocity_threshold)

    x, y, t = interp_filt_position(
        x, y, t, fs=sampling_rate, f_cut=low_pass_frequency)
    mask = t <= stop_time
    #mask = t2 <= stop_time
    x = x[mask]
    y = y[mask]
    t = t[mask]

    vel = np.gradient([x, y], axis=1) / np.gradient(t)
    speed = np.linalg.norm(vel, axis=0)

    return x, y, t, speed


def get_data_path(action):
    # action_path = action._backend.path
    # project_path = action_path.parent.parent
    #data_path = action.data['main']
    data_path = str(action.data_path('main'))

    # print("Project path: {}\nData path: {}".format(project_path, data_path))
    return data_path


def get_sample_rate(data_path, default_sample_rate=30000*pq.Hz):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    sr = default_sample_rate
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            if 'sample_rate' in ephys.attrs.keys():
                sr = ephys.attrs['sample_rate']
    return sr


def load_lfp(data_path, channel_group):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    # LFP
    t_stop = f.attrs['session_duration']
    _lfp = f['processing']['electrophysiology']['channel_group_{}'.format(channel_group)]['LFP']
    keys = list(_lfp.keys())
    electrode_value = [_lfp[key]['data'].value.flatten() for key in keys]
    electrode_idx = [_lfp[key].attrs['electrode_idx'] for key in keys]
    sampling_rate = _lfp[keys[0]].attrs['sample_rate']
    units = _lfp[keys[0]]['data'].attrs['unit']
    LFP = np.r_[[_lfp[key]['data'].value.flatten() for key in keys]].T
    LFP = LFP[:, np.argsort(electrode_idx)]

    LFP = neo.AnalogSignal(
        LFP, units=units, t_stop=t_stop, sampling_rate=sampling_rate,
        **{'electrode_idx': electrode_idx})
    LFP = LFP.rescale('mV')
    return LFP


def sort_by_cluster_id(spike_trains):
    if len(spike_trains) == 0:
        return spike_trains
    if 'name' not in spike_trains[0].annotations:
        print('Unable to get cluster_id, save with phy to create')
    sorted_sptrs = sorted(
        spike_trains,
        key=lambda x: int(x.annotations['name'].lower().replace('unit #', '')))
    return sorted_sptrs


def load_epochs(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    epochs_group = f['epochs']
    epochs = []
    for group in epochs_group.values():
        if 'timestamps' in group.keys():
            epo = _read_epoch(f, group.name)
            epochs.append(epo)
        else:
            for g in group.values():
                if 'timestamps' in g.keys():
                    epo = _read_epoch(f, g.name)
                    epochs.append(epo)
    return epochs


def get_channel_groups(data_path):
    '''
    Returns channeø groups of processing/electrophysiology
    Parameters
    ----------
    data_path: Path
        The action data path
    Returns
    -------
    channel groups: list
        The channel groups
    '''
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    channel_groups = []
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            for chname, ch in ephys.items():
                if 'channel' in chname:
                    channel_groups.append(int(chname.split('_')[-1]))
    return channel_groups


def load_spiketrains(data_path, channel_group=None, load_waveforms=False, t_start=0 * pq.s):
    '''
    Parameters
    ----------
    data_path
    channel_group
    load_waveforms
    remove_label
    Returns
    -------
    '''
    sample_rate = get_sample_rate(data_path)
    sorting = se.ExdirSortingExtractor(
        data_path, sample_rate=sample_rate,
        channel_group=channel_group, load_waveforms=load_waveforms)
    sptr = []
    # build neo pbjects
    for u in sorting.get_unit_ids():
        times = sorting.get_unit_spike_train(u) / sample_rate
        t_stop = np.max(times)
        times = times - t_start
        times = times[np.where(times > 0)]
        if load_waveforms and 'waveforms' in sorting.get_unit_spike_feature_names(u):
            wf = sorting.get_unit_spike_features(u, 'waveforms')
            wf = wf[np.where(times > 0)] * pq.uV
        else:
            wf = None
        st = neo.SpikeTrain(times=times, t_stop=t_stop, waveforms=wf, sampling_rate=sample_rate)
        for p in sorting.get_unit_property_names(u):
            st.annotations.update({p: sorting.get_unit_property(u, p)})
        sptr.append(st)

    return sptr


def load_unit_annotations(data_path, channel_group):
    '''
    Parameters
    ----------
    data_path
    channel_group
    Returns
    -------
    '''
    sample_rate = get_sample_rate(data_path)
    sorting = se.ExdirSortingExtractor(
        data_path, sample_rate=sample_rate,
        channel_group=channel_group, load_waveforms=False)
    units = []
    for u in sorting.get_unit_ids():
        annotations = {}
        for p in sorting.get_unit_property_names(u):
            annotations.update({p: sorting.get_unit_property(u, p)})
        units.append(annotations)
    return units


def load_spike_train(data_path, channel_id, unit_id):
    root_group = exdir.File(data_path, "r", plugins=[exdir.plugins.quantities,
                                                exdir.plugins.git_lfs])
    u_path = unit_path(channel_id, unit_id)
    unit_group = root_group[u_path]
    # spiketrain data
    sptr_group = unit_group
    metadata = {}
    times = np.array(sptr_group['times'].data)

    t_stop = sptr_group.parent.attrs['stop_time']
    t_start = sptr_group.parent.attrs['start_time']
    metadata.update(sptr_group['times'].attrs.to_dict())
    metadata.update({'exdir_path': str(data_path)})
    sptr = neo.SpikeTrain(times=times, units = 's',
                      t_stop=t_stop,
                      t_start=t_start,
                      waveforms=None,
                      sampling_rate=None,
                      **metadata)
    return sptr
#-------------------------------------------------------------------------------
    # def read_analogsignal(self, path, cascade=True, lazy=False):
    #     channel_group = self._exdir_directory[path]
    #     group_id = channel_group.attrs['electrode_group_id']
    #
    #     for lfp_group in channel_group['LFP'].values():
    #             ana = self.read_analogsignal(lfp_group.name,
    #                                          cascade=cascade,
    #                                          lazy=lazy)
    #             chx.analogsignals.append(ana)
    #             ana.channel_index = chx
    #
    #     group = self._exdir_directory[path]
    #     signal = group["data"]
    #     attrs = {'exdir_path': path}
    #     attrs.update(group.attrs.to_dict())
    #     ana = AnalogSignal(signal.data,
    #                            units=signal.attrs["unit"],
    #                            sampling_rate=group.attrs['sample_rate'],
    #                            **attrs)
    #     return ana
