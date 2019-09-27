from .track_units_tools import make_dissimilary_matrix, compute_templates, make_possible_match, make_best_match, \
    make_hungarian_match, get_unit_id, dissimilarity
from .data_processing import get_data_path, load_spiketrains, get_channel_groups, load_unit_annotations
import matplotlib.pylab as plt
import numpy as np


class TrackingSession:
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """

    def __init__(self, action_id_1, action_id_2, actions, channel_group=None,
                 max_dissimilarity=10, dissimilarity_function=None, verbose=False):

        data_path_1 = get_data_path(actions[action_id_1])
        data_path_2 = get_data_path(actions[action_id_2])

        self._actions = actions
        self.action_id_1 = action_id_1
        self.action_id_2 = action_id_2
        self._channel_group = channel_group
        self.name_list = [action_id_1, action_id_2]
        self.max_dissimilarity = max_dissimilarity
        self.dissimilarity_function = dissimilarity_function
        self._verbose = verbose

        if channel_group is None:
            channel_groups = get_channel_groups(data_path_1)
            self.matches = {}
            for chan in channel_groups:
                self.matches[chan] = dict()
        else:
            self.matches = {channel_group: dict()}

        for chan_grp in self.matches.keys():
            unit_annotations_1 = load_unit_annotations(
                data_path_1, channel_group=chan_grp)
            unit_annotations_2 = load_unit_annotations(
                data_path_2, channel_group=chan_grp)

            unit_ids_1 = []
            unit_ids_2 = []
            for st in unit_annotations_1:
                unit_ids_1.append(get_unit_id(st))
            for st in unit_annotations_2:
                unit_ids_2.append(get_unit_id(st))

            self.matches[chan_grp]['unit_ids_1'] = np.array(unit_ids_1)
            self.matches[chan_grp]['unit_ids_2'] = np.array(unit_ids_2)

            if len(unit_annotations_1) > 0 and len(unit_annotations_2) > 0:

                self._do_dissimilarity(chan_grp)
                self._do_matching(chan_grp)

    def waveforms_1(self, channel_group):
        action_1 = self._actions[self.action_id_1]

        data_path_1 = get_data_path(action_1)

        spike_trains_1 = load_spiketrains(
            data_path_1, channel_group=channel_group, load_waveforms=True)

        return [np.array(sptr.waveforms) for sptr in spike_trains_1]

    def waveforms_2(self, channel_group):
        action_2 = self._actions[self.action_id_2]

        data_path_2 = get_data_path(action_2)

        spike_trains_2 = load_spiketrains(
            data_path_2, channel_group=channel_group, load_waveforms=True)

        return [np.array(sptr.waveforms) for sptr in spike_trains_2]

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
        self.matches[chan_group]['dissimilarity_scores'] = make_dissimilary_matrix(
            self.waveforms_1(chan_group),
            self.waveforms_2(chan_group),
            self.matches[chan_group]['unit_ids_1'],
            self.matches[chan_group]['unit_ids_2'],
            function=self.dissimilarity_function)

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
                    i1 = np.where(self.matches[ch_group]['unit_ids_1'] == match12.index[i])
                    i2 = np.where(self.matches[ch_group]['unit_ids_2'] == j)
                    template1 = np.squeeze(
                        compute_templates(
                            self.matches[ch_group]['waveforms_1'][i1])).T
                    ax_list[0].plot(template1, color='C0')
                    ax_list[0].set_title('Unit ' + str(match12.index[i]))
                    template2 = np.squeeze(
                        compute_templates(
                            self.matches[ch_group]['waveforms_2'][i1])).T
                    ax_list[1].plot(template2, color='C0')
                    ax_list[1].set_title('Unit ' + str(j))
                    ax_list[0].set_ylabel(self.name_list[0])
                    ax_list[1].set_ylabel(self.name_list[1])
                    ax_list[0].set_ylim(ylim)
                    ax_list[1].set_ylim(ylim)
                else:
                    id_ax = 0
                    for i, j in enumerate(match12):
                        if j != -1:
                            i1 = np.where(self.matches[ch_group]['unit_ids_1'] == match12.index[i])
                            i2 = np.where(self.matches[ch_group]['unit_ids_2'] == j)

                            if id_ax == 0:
                                ax_list[0, id_ax].set_ylabel(self.name_list[0])
                                ax_list[1, id_ax].set_ylabel(self.name_list[1])
                            template1 = np.squeeze(
                                compute_templates(
                                    self.matches[ch_group]['waveforms_1'][i1])).T
                            ax_list[0, id_ax].plot(template1, color='C'+str(id_ax))
                            ax_list[0, id_ax].set_title('Unit ' + str(match12.index[i]))
                            template2 = np.squeeze(
                                compute_templates(
                                    self.matches[ch_group]['waveforms_2'][i1])).T
                            ax_list[1, id_ax].plot(template2, color='C'+str(id_ax))
                            ax_list[1, id_ax].set_title('Unit ' + str(j))
                            ax_list[0, id_ax].set_ylim(ylim)
                            ax_list[1, id_ax].set_ylim(ylim)
                            id_ax += 1
            else:
                print('No matched units for group', ch_group)
                continue
