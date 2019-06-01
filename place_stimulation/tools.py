# This is work in progress,
import neo
import numpy as np
import exdir
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib
import quantities as pq
import spikeextractors as se
import os


def get_data_path(action):
    action_path = action._backend.path
    project_path = action_path.parent.parent
    #data_path = action.data['main']
    data_path = str(pathlib.Path(pathlib.PureWindowsPath(action.data['main'])))

    print("Project path: {}\nData path: {}".format(project_path, data_path))
    return project_path / data_path


def get_sample_rate(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    sr = None
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            sr = ephys.attrs['sample_rate']
    return sr


def load_lfp(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    # LFP
    t_stop = f.attrs['session_duration']
    _lfp = f['processing']['electrophysiology']['channel_group_0']['LFP']
    keys = list(_lfp.keys())
    electrode_value = [_lfp[key]['data'].value.flatten() for key in keys]
    electrode_idx = [_lfp[key].attrs['electrode_idx'] for key in keys]
    sampling_rate = _lfp[keys[0]].attrs['sample_rate']
    units = _lfp[keys[0]]['data'].attrs['unit']
    LFP = np.r_[[_lfp[key]['data'].value.flatten() for key in keys]].T
    #LFP = (LFP.T - np.median(np.array(LFP), axis=-1)).T #CMR reference
    #LFP = (LFP.T - LFP[:, 0]).T # use topmost channel as reference
    LFP = LFP[:, np.argsort(electrode_idx)]

    LFP = neo.AnalogSignal(LFP,
                           units=units, t_stop=t_stop, sampling_rate=sampling_rate)
    LFP = LFP.rescale('mV')
    return LFP


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
    # io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])
    # blk = io.read_block()
    # seg = blk.segments[0]
    # epochs = seg.epochs
    return epochs


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
    sorting = se.ExdirSortingExtractor(data_path, sample_rate=sample_rate.magnitude,
                                       channel_group=channel_group, load_waveforms=load_waveforms)
    sptr = []
    # build neo pbjects
    for u in sorting.get_unit_ids():
        times = sorting.get_unit_spike_train(u) / sample_rate
        t_stop = np.max(times)
        if load_waveforms and 'waveforms' in sorting.get_unit_spike_feature_names(u):
            wf = sorting.get_unit_spike_features(u, 'waveforms')
        else:
            wf = None
        times = times - t_start
        times = times[np.where(times > 0)]
        wf = wf[np.where(times > 0)]
        st = neo.SpikeTrain(times=times, t_stop=t_stop, waveforms=wf)
        st.annotate(channel_group=sorting.get_unit_property(u, 'group'))
        sptr.append(st)

    return sptr


def load_tracking(data_path, select_tracking=None, interp=False, reverse_y=True, fc=5*pq.Hz, t_start=0 * pq.s):
    '''

    Parameters
    ----------
    data_path
    par
    select_tracking
    interp
    fc
    t_start

    Returns
    -------

    '''
    root_group = exdir.File(str(data_path), plugins=[exdir.plugins.quantities,
                                                     exdir.plugins.git_lfs])
    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    stop_time = position_group.attrs.to_dict()["stop_time"]
    led0 = False
    led1 = False

    if 'led_0' in position_group.keys():
        x1, y1 = position_group['led_0']['data'].data.T
        t1 = position_group['led_0']['timestamps'].data
        unit = t1.units
        x1, y1, t1 = rm_nans(x1, y1, t1)
        t1 = t1 * unit
        x1, y1, t1 = rm_inconsistent_timestamps(x1, y1, t1)
        led0 = True
    if 'led_1' in position_group.keys():
        x2, y2 = position_group['led_1']['data'].data.T
        t2 = position_group['led_1']['timestamps'].data
        unit = t2.units
        x2, y2, t2 = rm_nans(x2, y2, t2)
        t2 = t2 * unit
        led1 = True

    if select_tracking is None and led0 and led1:
        x, y, t = select_best_position(x1, y1, t1, x2, y2, t2)
    elif select_tracking is None and led0:
        x, y, t = x1, y1, t1
    elif select_tracking is None and led1:
        x, y, t = x2, y2, t2
    elif select_tracking == 0 and led0:
        x, y, t = x1, y1, t1
    elif select_tracking == 1 and led1:
        x, y, t = x2, y2, t2
    else:
        raise Exception('Selected tracking not found')
    # t = t * unit

    dt = np.mean(np.diff(t))
    fs = 1. / dt
    print(fs)

    # remove zeros
    idx_non_zero_x = np.where(x != 0)
    xf, yf, tf = x[idx_non_zero_x], y[idx_non_zero_x], t[idx_non_zero_x]
    idx_non_zero_y = np.where(yf != 0)
    xf, yf, tf = xf[idx_non_zero_y], yf[idx_non_zero_y], tf[idx_non_zero_y]

    print("Removed", (len(x) - len(xf)) / len(x) * 100, '% of tracking samples')

    if interp:
        xf, yf, tf = interp_filt_position(xf, yf, tf, pos_fs=fs, f_cut=fc)
    # mask = t <= stop_time
    # x = x[mask]
    # y = y[mask]
    # t = t[mask]
    tf = tf - t_start
    idxs = np.where(tf > 0)
    tf = tf[idxs]
    xf = xf[idxs]
    yf = yf[idxs]

    vel = np.gradient([xf, yf], axis=1)/dt
    speed = np.linalg.norm(vel, axis=0)

    if reverse_y:
        yf = np.max(yf) - yf

    xf, yf, tf = rm_inconsistent_timestamps(xf, yf, tf)

    return xf, yf, tf, speed


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


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return out


def fftcorrelate2d(arr1, arr2, mode='full', normalize=False):
    from scipy.signal import fftconvolve
    if normalize:
        a_ = np.reshape(arr1, (1, arr1.size))
        v_ = np.reshape(arr2, (1, arr2.size))
        arr1 = (arr1 - np.mean(a_)) / (np.std(a_) * len(a_))
        arr2 = (arr2 - np.mean(v_)) / np.std(v_)
    corr = fftconvolve(arr1, np.fliplr(np.flipud(arr2)), mode=mode)
    return corr


def velocity_threshold(x, y, t, threshold):
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
    unit = t.units
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    r = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    v = np.divide(r, np.diff(t))
    speed_lim = np.concatenate(([False], v > threshold), axis=0)
    x[speed_lim] = np.nan
    y[speed_lim] = np.nan
    x, y, t = rm_nans(x, y, t)
    t = t * unit
    return x, y, t


def select_best_position(x1, y1, t1, x2, y2, t2, speed_filter=5):
    """
    selects position data with least nan after speed filtering
    Parameters
    ----------
    x1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    y1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    t1 : quantities.Quantity array in s
        1d vector of times from LED 1 at x, y positions
    x2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    y2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    t2 : quantities.Quantity array in s
        1d vector of times from LED 2 at x, y positions
    speed_filter : None or quantities in m/s
        threshold filter for translational speed
    """
    x1, y1, t1, x2, y2, t2 = _cut_to_same_len(x1, y1, t1, x2, y2, t2)
    measurements1 = len(x1)
    measurements2 = len(x2)
    # x1, y1, t1 = rm_nans(x1, y1, t1)
    # x2, y2, t2 = rm_nans(x2, y2, t2)
    if speed_filter is not None:
        x1, y1, t1 = velocity_threshold(x1, y1, t1, speed_filter)
        x2, y2, t2 = velocity_threshold(x2, y2, t2, speed_filter)

    if len(x1) > len(x2):
        print('Removed %.2f %% invalid measurements in path' %
              ((1. - len(x1) / float(measurements1)) * 100.))
        x = x1
        y = y1
        t = t1
    else:
        print('Removed %.2f %% invalid measurements in path' %
              ((1. - len(x2) / float(measurements2)) * 100.))
        x = x2
        y = y2
        t = t2
    return x, y, t


def interp_filt_position(x, y, tm, box_xlen=1 , box_ylen=1 ,
                         pos_fs=100, f_cut=10):
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
    pos_fs : quantities scalar in Hz
        return radians
    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    pos_unit = x.units
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'

    if not isinstance(pos_fs, pq.Quantity):
        t = np.arange(tm.min(), tm.max() + 1. / pos_fs * tm.units, 1. / pos_fs * tm.units)
    else:
        t = np.arange(tm.min(), tm.max() + 1. / pos_fs, 1. / pos_fs)
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filtering
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(2, f_cut * 2 / pos_fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))
    if x.min() < 0 or x.max() > box_xlen or y.min() < 0 or y.max() > box_ylen:
        print(
            "WARNING! Interpolation produces path values " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))
        x[x < 0] = 0
        x[x > box_xlen] = 0
        y[y < 0] = 0
        y[y > box_ylen] = 0

    R = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    V = R / np.diff(t)
    print('Maximum speed {}'.format(V.max()))
    t = t * tm.units
    x = x * pos_unit
    y = y * pos_unit
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


def rm_inconsistent_timestamps(x, y, t):
    """
    Removes timestamps not linearly increasing
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    Returns
    -------
    x : quantities.Quantity array in m
        1d vector of cleaned x positions
    y : quantities.Quantity array in m
        1d vector of cleaned y positions
    t : quantities.Quantity array in s
        1d vector of cleaned times at x, y positions
    """
    diff_violations = np.where(np.diff(t) <= 0)[0]
    unit_t = t.units
    unit_pos = x.units
    if len(diff_violations) > 0:
        print('Timestamps diff violations:', len(diff_violations))
        tc = np.delete(t, diff_violations + 1) * unit_t
        xc = np.delete(x, diff_violations + 1) * unit_pos
        yc = np.delete(y, diff_violations + 1) * unit_pos
    else:
        tc = t
        xc = x
        yc = y
    return xc, yc, tc
