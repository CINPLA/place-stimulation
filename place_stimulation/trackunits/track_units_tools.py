import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def dissimilarity(template1, template2):
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
    t_i_lin = template1.reshape(template1.shape[0] * template1.shape[1])
    t_j_lin = template2.reshape(template2.shape[0] * template2.shape[1])

    max_val = np.max([np.max(np.abs(t_i_lin)), np.max(np.abs(t_j_lin))])

    return np.sum(np.abs(t_i_lin / max_val - t_j_lin / max_val)) / (template1.shape[0] * template1.shape[1])


def compute_templates(sptr):
    '''

    Parameters
    ----------
    sptr

    Returns
    -------

    '''
    templates = []
    for st in sptr:
        t = np.mean(st.waveforms, axis=0)
        templates.append(t)

    return np.array(templates)


def make_dissimilary_matrix(templates1, templates2, chan_idx1=None, chan_idx2=None, units1=None, units2=None):
    if chan_idx1 is None and chan_idx2 is None:
        assert templates1.shape == templates2.shape
        chans1 = np.arange(templates1.shape[1])
        chans2 = np.arange(templates2.shape[1])
    else:
        if len(chan_idx1) > len(chan_idx2):
            all_chans = np.arange(templates1.shape[1])
            chans = []
            for c in all_chans:
                if (chan_idx1 - np.min(chan_idx1))[c] in (chan_idx2 - np.min(chan_idx2)):
                    chans.append(c)
            chans1 = np.array(chans)
            chans2 = np.arange(templates2.shape[1])
        elif len(chan_idx1) < len(chan_idx2):
            all_chans = np.arange(templates2.shape[1])
            chans = []
            for c in all_chans:
                if (chan_idx2 - np.min(chan_idx2))[c] in (chan_idx1 - np.min(chan_idx1)):
                    chans.append(c)
            chans2 = np.array(chans)
            chans1 = np.arange(templates1.shape[1])
        else:
            chans1 = chans2 = np.arange(templates1.shape[1])

    diss_matrix = np.zeros((templates1.shape[0], templates2.shape[0]))

    for i, t1 in enumerate(templates1):
        for j, t2 in enumerate(templates2):
            diss_matrix[i, j] = dissimilarity(t1[chans1], t2[chans2])

    diss_matrix = pd.DataFrame(diss_matrix, index=units1,
                               columns=units2)

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
        ind_max = np.argmax(scores[i1, :])
        if scores[i1, ind_max] <= max_dissimilarity:
            best_match_12[u1] = unit2_ids[ind_max]
        else:
            best_match_12[u1] = -1

    best_match_21 = pd.Series(index=unit2_ids, dtype='int64')
    for i2, u2 in enumerate(unit2_ids):
        ind_max = np.argmax(scores[:, i2])
        if scores[ind_max, i2] <= max_dissimilarity:
            best_match_21[u2] = unit1_ids[ind_max]
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
    scores[scores > max_dissimilarity] = max_dissimilarity

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
