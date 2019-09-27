import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from matplotlib import gridspec
import matplotlib.pyplot as plt


def get_unit_id(unit):
    try:
        uid = int(unit.annotations['name'].split('#')[-1])
    except AttributeError:
        uid = int(unit['name'].split('#')[-1])
    return uid


def dissimilarity(waveforms1, waveforms2):
    """
    Returns a value of dissimilarity of the mean between two or more
    spike templates.
    Parameters
    ----------
    templates : list object (see Notes)
        List containing the mean waveform over each electrode of spike sorted
        spiketrains from at least one electrode. All elements in the list must
        be of equal size, that is, the number of electrodes must be equal, and
        the number of points on the waveform must be equal.
    Returns
    -------
    diss : numpy array-like
        Returns a matrix containing the computed dissimilarity between the mean
        of the spiketrain, for the same channel.
    """
    template1 = compute_template(waveforms1)
    template2 = compute_template(waveforms2)

    t_i_lin = template1.ravel()
    t_j_lin = template2.ravel()

    max_val = np.max([np.max(np.abs(template1)), np.max(np.abs(template2))])

    return np.mean(np.abs(t_i_lin / max_val - t_j_lin / max_val))


def dissimilarity_weighted(waveforms1, waveforms2):
    """
    Returns a value of dissimilarity of the mean between two or more
    spike templates.
    Parameters
    ----------
    templates : list object (see Notes)
        List containing the mean waveform over each electrode of spike sorted
        spiketrains from at least one electrode. All elements in the list must
        be of equal size, that is, the number of electrodes must be equal, and
        the number of points on the waveform must be equal.
    Returns
    -------
    diss : numpy array-like
        Returns a matrix containing the computed dissimilarity between the mean
        of the spiketrain, for the same channel.
    """
    std1 = np.std(waveforms1, axis=0)
    template1 = np.mean(waveforms1, axis=0)# * std1
    std2 = np.std(waveforms2, axis=0)
    template2 = np.mean(waveforms2, axis=0)# * std2

    t_i_lin = (template1 * std1).ravel()
    t_j_lin = (template2 * std2).ravel()
    # t_i_lin = template1.ravel()
    # t_j_lin = template2.ravel()

    max_val = np.max([np.max(np.abs(template1)), np.max(np.abs(template2))])

    return np.mean(np.abs(t_i_lin / max_val - t_j_lin / max_val))


def compute_template(waveforms):
    return np.mean(waveforms, axis=0)


def get_waveform(spike_train):
    try:
        return spike_train.waveforms
    except AttributeError:
        return spike_train


def compute_templates(spike_trains):
    templates = [compute_template(get_waveform(st)) for st in spike_trains]
    return np.array(templates)


def make_dissimilary_matrix(waveforms1, waveforms2, unit_ids1, unit_ids2, function=None):
    function = dissimilarity if function is None else function
    diss_matrix = np.zeros((len(waveforms1), len(waveforms2)))

    for i, w1 in enumerate(waveforms1):
        for j, w2 in enumerate(waveforms2):
            diss_matrix[i, j] = function(w1, w2)

    diss_matrix = pd.DataFrame(
        diss_matrix, index=unit_ids1, columns=unit_ids2)

    return diss_matrix


def make_possible_match(dissimilarity_scores, max_dissimilarity):
    """
    Given an agreement matrix and a max_dissimilarity threhold.
    Return as a dict all possible match for each spiketrain in each side.

    Note : this is symmetric.


    Parameters
    ----------
    dissimilarity_scores: pd.DataFrame

    max_dissimilarity: float


    Returns
    -----------
    best_match_12: pd.Series

    best_match_21: pd.Series

    """
    unit1_ids = np.array(dissimilarity_scores.index)
    unit2_ids = np.array(dissimilarity_scores.columns)

    # threhold the matrix
    scores = dissimilarity_scores.values.copy()
    scores[scores > max_dissimilarity] = np.inf

    possible_match_12 = {}
    for i1, u1 in enumerate(unit1_ids):
        inds_match = np.isfinite(scores[i1, :])
        possible_match_12[u1] = unit2_ids[inds_match]

    possible_match_21 = {}
    for i2, u2 in enumerate(unit2_ids):
        inds_match = np.isfinite(scores[:, i2])
        possible_match_21[u2] = unit1_ids[inds_match]

    return possible_match_12, possible_match_21


def make_best_match(dissimilarity_scores, max_dissimilarity):
    """
    Given an agreement matrix and a max_dissimilarity threhold.
    return a dict a best match for each units independently of others.

    Note : this is symmetric.

    Parameters
    ----------
    dissimilarity_scores: pd.DataFrame

    max_dissimilarity: float


    Returns
    -----------
    best_match_12: pd.Series

    best_match_21: pd.Series


    """
    unit1_ids = np.array(dissimilarity_scores.index)
    unit2_ids = np.array(dissimilarity_scores.columns)

    scores = dissimilarity_scores.values.copy()

    best_match_12 = pd.Series(index=unit1_ids, dtype='int64')
    for i1, u1 in enumerate(unit1_ids):
        ind_min = np.argmin(scores[i1, :])
        if scores[i1, ind_min] <= max_dissimilarity:
            best_match_12[u1] = unit2_ids[ind_min]
        else:
            best_match_12[u1] = -1

    best_match_21 = pd.Series(index=unit2_ids, dtype='int64')
    for i2, u2 in enumerate(unit2_ids):
        ind_min = np.argmin(scores[:, i2])
        if scores[ind_min, i2] <= max_dissimilarity:
            best_match_21[u2] = unit1_ids[ind_min]
        else:
            best_match_21[u2] = -1

    return best_match_12, best_match_21


def make_hungarian_match(dissimilarity_scores, max_dissimilarity):
    """
    Given an agreement matrix and a max_dissimilarity threhold.
    return the "optimal" match with the "hungarian" algo.
    This use internally the scipy.optimze.linear_sum_assignment implementation.

    Parameters
    ----------
    dissimilarity_scores: pd.DataFrame

    max_dissimilarity: float


    Returns
    -----------
    hungarian_match_12: pd.Series

    hungarian_match_21: pd.Series

    """
    unit1_ids = np.array(dissimilarity_scores.index)
    unit2_ids = np.array(dissimilarity_scores.columns)

    # threhold the matrix
    scores = dissimilarity_scores.values.copy()

    [inds1, inds2] = linear_sum_assignment(scores)

    hungarian_match_12 = pd.Series(index=unit1_ids, dtype='int64')
    hungarian_match_12[:] = -1
    hungarian_match_21 = pd.Series(index=unit2_ids, dtype='int64')
    hungarian_match_21[:] = -1

    for i1, i2 in zip(inds1, inds2):
        u1 = unit1_ids[i1]
        u2 = unit2_ids[i2]
        if dissimilarity_scores.at[u1, u2] < max_dissimilarity:
            hungarian_match_12[u1] = u2
            hungarian_match_21[u2] = u1

    return hungarian_match_12, hungarian_match_21


def lighten_color(color, amount=0.7):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_waveform(wf, fig, gs, axs=None, **kwargs):

    nrc = wf.shape[1]
    if axs is None:
        gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
        axs = [fig.add_subplot(gs0[0])]
        axs.extend([fig.add_subplot(gs0[i], sharey=axs[0], sharex=axs[0]) for i in range(1, nrc)])
    for c in range(nrc):
        wf_ch = wf[:, c, :]
        m = np.mean(wf_ch, axis=0)
        sd = np.std(wf_ch, axis=0) / wf.shape[0]
        samples = np.arange(wf.shape[2])
        axs[c].plot(samples, m, **kwargs)
        axs[c].fill_between(samples, m-sd, m+sd, alpha=.1, color=kwargs.get('color'))
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    return axs
