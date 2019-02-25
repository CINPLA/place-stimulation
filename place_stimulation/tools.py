# This is work in progress,
import neo
import numpy as np
import exdir
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib


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
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    r = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    v = np.divide(r, np.diff(t))
    speed_lim = np.concatenate(([False], v > threshold), axis=0)
    x[speed_lim] = np.nan
    y[speed_lim] = np.nan
    x, y, t = rm_nans(x, y, t)
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
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)
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
                         pos_fs=100 , f_cut=10 ):
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
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    t = np.arange(tm.min(), tm.max() + 1. / pos_fs * tm.units, 1. / pos_fs * tm.units)
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


def load_tracking(data_path, par, select_tracking=None, interp=False):
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
        x1, y1, t1 = rm_nans(x1, y1, t1)
        unit = t1.units
        led0 = True
    if 'led_1' in position_group.keys():
        x2, y2 = position_group['led_1']['data'].data.T
        t2 = position_group['led_1']['timestamps'].data
        x2, y2, t2 = rm_nans(x2, y2, t2)
        unit = t2.units
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
    t = t * unit

    if interp:
        x, y, t = interp_filt_position(x, y, t, pos_fs=par['pos_fs'], f_cut=par['f_cut'])
    # mask = t <= stop_time
    # x = x[mask]
    # y = y[mask]
    # t = t[mask]

    # remove zeros
    idx_non_zero = np.where(x != 0)
    x, y, t = x[idx_non_zero], y[idx_non_zero], t[idx_non_zero]
    idx_non_zero = np.where(y != 0)
    x, y, t = x[idx_non_zero], y[idx_non_zero], t[idx_non_zero]

    dt = np.mean(np.diff(t))
    vel = np.gradient([x, y], axis=1)/dt
    speed = np.linalg.norm(vel, axis=0)

    return x, y, t, speed


def load_spiketrains(data_path, channel_idx=None, remove_label='noise'):
    io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs.Plugin(verbose=True)])
    if channel_idx is None:
        blk = io.read_block()
        sptr = blk.segments[0].spiketrains
    else:
        blk = io.read_block(channel_group_idx=channel_idx)
        channels = blk.channel_indexes
        chx = channels[0]
        sptr = [u.spiketrains[0] for u in chx.units]
    if remove_label is not None:
        sptr = [s for s in sptr if remove_label not in s.annotations['cluster_group']]
    return sptr


def load_epochs(data_path):
    io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])
    blk = io.read_block(channel_group_idx=0)
    seg = blk.segments[0]
    epochs = seg.epochs
    return epochs


def get_data_path(action):
    action_path = action._backend.path
    project_path = action_path.parent.parent
    print(project_path)
    # data_path = action.data['main']
    data_path = str(pathlib.Path(pathlib.PureWindowsPath(action.data['main'])))
    print(data_path)
    return project_path / data_path
