from .track_units_tools import make_dissimilary_matrix, compute_templates, make_possible_match, make_best_match, \
    make_hungarian_match
from ..tools import get_data_path, load_spiketrains, get_channel_groups
import matplotlib.pylab as plt
import numpy as np


class TrackingSession:
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """

    def __init__(self, action1, action2, actions, channel_group=None,
                 max_dissimilarity=10, verbose=False):

        act1 = actions[action1]
        act2 = actions[action2]

        dp1 = get_data_path(act1)
        dp2 = get_data_path(act2)

        self.action1 = action1
        self.action2 = action2
        self._channel_group = channel_group
        self.name_list = [action1, action2]
        self.max_dissimilarity = max_dissimilarity
        self._verbose = verbose

        if channel_group is None:
            channel_groups = get_channel_groups(dp1)
            self.matches = {}
            for chan in channel_groups:
                self.matches[chan] = dict()
        else:
            self.matches = {channel_group: dict()}

        for chan_grp in self.matches.keys():
            sptr1 = load_spiketrains(dp1, channel_group=chan_grp, load_waveforms=True)
            sptr2 = load_spiketrains(dp2, channel_group=chan_grp, load_waveforms=True)

            self.matches[chan_grp]['templates1'] = compute_templates(sptr1)
            self.matches[chan_grp]['templates2'] = compute_templates(sptr2)

            if len(self.matches[chan_grp]['templates1']) > 0 and len(self.matches[chan_grp]['templates2']) > 0:
                self.matches[chan_grp]['chan_idx1'] = sptr1[0].annotations['channel_idx']
                self.matches[chan_grp]['chan_idx2'] = sptr2[0].annotations['channel_idx']

                units1 = []
                units2 = []
                for st in sptr1:
                    units1.append(st.annotations['unit_id'])
                for st in sptr2:
                    units2.append(st.annotations['unit_id'])

                self.matches[chan_grp]['units1'] = np.array(units1)
                self.matches[chan_grp]['units2'] = np.array(units2)

                self._do_dissimilarity(chan_grp)
                self._do_matching(chan_grp)
            else:
                self.matches[chan_grp]['units1'] = np.array([])
                self.matches[chan_grp]['units2'] = np.array([])


    @property
    def session1_name(self):
        return self.name_list[0]

    @property
    def session2_name(self):
        return self.name_list[1]

    def _do_dissimilarity(self, chan_group):
        if self._verbose:
            print('Agreement scores...')

        # agreement matrix score for each pair
        self.matches[chan_group]['dissimilarity_scores'] = make_dissimilary_matrix(self.matches[chan_group][
                                                                                       'templates1'],
                                                                                   self.matches[chan_group][
                                                                                       'templates2'],
                                                                                   self.matches[chan_group][
                                                                                       'chan_idx1'],
                                                                                   self.matches[chan_group][
                                                                                       'chan_idx2'],
                                                                                   self.matches[chan_group][
                                                                                       'units1'],
                                                                                   self.matches[chan_group][
                                                                                       'units2'])

    def _do_matching(self, chan_group):
        # must be implemented in subclass
        if self._verbose:
            print("Matching...")

        self.matches[chan_group]['possible_match_12'], self.matches[chan_group]['possible_match_21'] = \
            make_possible_match(self.matches[chan_group]['dissimilarity_scores'], self.max_dissimilarity)
        self.matches[chan_group]['best_match_12'], self.matches[chan_group]['best_match_21'] = \
            make_best_match(self.matches[chan_group]['dissimilarity_scores'], self.max_dissimilarity)
        self.matches[chan_group]['hungarian_match_12'], self.matches[chan_group]['hungarian_match_21'] = \
            make_hungarian_match(self.matches[chan_group]['dissimilarity_scores'], self.max_dissimilarity)

    def plot_matched_units(self, match_mode='hungarian', chan_group=None, ylim=[-200, 50], figsize=(15, 15)):
        '''

        Parameters
        ----------
        match_mode

        Returns
        -------

        '''
        if chan_group is None:
            ch_groups = self.matches.keys()
        else:
            ch_groups = [chan_group]

        for ch_group in ch_groups:
            if 'hungarian_match_12' not in self.matches[ch_group].keys():
                print('Not units for group', ch_group)
                continue

            if match_mode == 'hungarian':
                match12 = self.matches[ch_group]['hungarian_match_12']
            elif match_mode == 'best':
                match12 = self.matches[ch_group]['best_match_12']

            num_matches = len(np.where(match12 != -1)[0])

            if num_matches > 0:

                fig, ax_list = plt.subplots(nrows=2, ncols=num_matches, figsize=figsize)
                fig.suptitle('Channel group ' + str(ch_group))

                if num_matches == 1:
                    i = np.where(match12 != -1)[0][0]
                    j = match12.iloc[i]
                    i1 = np.where(self.matches[ch_group]['units1'] == match12.index[i])
                    i2 = np.where(self.matches[ch_group]['units2'] == j)
                    ax_list[0].plot(np.squeeze(self.matches[ch_group]['templates1'][i1]).T, color='C0')
                    ax_list[0].set_title('Unit ' + str(match12.index[i]))
                    ax_list[1].plot(np.squeeze(self.matches[ch_group]['templates2'][i2]).T, color='C1')
                    ax_list[1].set_title('Unit ' + str(j))
                    ax_list[0].set_ylabel(self.name_list[0])
                    ax_list[1].set_ylabel(self.name_list[1])
                    ax_list[0].set_ylim(ylim)
                    ax_list[1].set_ylim(ylim)
                else:
                    id_ax = 0
                    for i, j in enumerate(match12):
                        if j != -1:
                            i1 = np.where(self.matches[ch_group]['units1'] == match12.index[i])
                            i2 = np.where(self.matches[ch_group]['units2'] == j)

                            if id_ax == 0:
                                ax_list[0, id_ax].set_ylabel(self.name_list[0])
                                ax_list[1, id_ax].set_ylabel(self.name_list[1])

                            ax_list[0, id_ax].plot(np.squeeze(self.matches[ch_group]['templates1'][i1]).T,
                                              color='C'+str(id_ax))
                            ax_list[0, id_ax].set_title('Unit ' + str(match12.index[i]))
                            ax_list[1, id_ax].plot(np.squeeze(self.matches[ch_group]['templates2'][i2]).T,
                                              color='C'+str(id_ax))
                            ax_list[1, id_ax].set_title('Unit ' + str(j))
                            ax_list[0, id_ax].set_ylim(ylim)
                            ax_list[1, id_ax].set_ylim(ylim)
                            id_ax += 1
            else:
                print('Not matched units for group', ch_group)
                continue
