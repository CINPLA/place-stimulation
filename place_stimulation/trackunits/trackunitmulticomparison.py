import numpy as np
import networkx as nx
from .trackunitcomparison import TrackingSession
from ..tools import get_data_path, get_channel_groups, load_spiketrains
import matplotlib.pylab as plt


class TrackMultipleSessions:
    def __init__(self, action_list, actions, channel_group=None,
                 max_dissimilarity=10, verbose=False):

        self.action_list = action_list
        self._actions = actions
        self._channel_group = channel_group
        self.max_dissimilarity = max_dissimilarity
        self._verbose = verbose

        self._do_matching(verbose)

    def get_action_list(self):
        return self.action_list

    def _do_matching(self, verbose):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step1: pairwise comparison')

        self.comparisons = []
        for i in range(len(self.action_list)):
            for j in range(i + 1, len(self.action_list)):
                if verbose:
                    print("  Comparing: ", self.action_list[i], " and ", self.action_list[j])
                comp = TrackingSession(self.action_list[i], self.action_list[j],
                                       actions=self._actions,
                                       max_dissimilarity=self.max_dissimilarity,
                                       channel_group=self._channel_group,
                                       verbose=False)
                self.comparisons.append(comp)

        if self._verbose:
            print('Multicomaprison step2: make graph')

        self.graphs = dict()
        self.all_units = dict()
        if self._channel_group is None:
            dp = get_data_path(self._actions[self.action_list[0]])
            channel_groups = get_channel_groups(dp)
        else:
            channel_groups = [self._channel_group]

        for ch in channel_groups:
            if self._verbose:
                print('Processing channel', ch)
            self.graphs[ch] = nx.Graph()

            # nodes
            for i, action in enumerate(self.action_list):
                dp = get_data_path(self._actions[action])
                sptr = load_spiketrains(data_path=dp, channel_group=ch)
                for st in sptr:
                    node_name = str(action) + '_' + str(st.annotations['unit_id'])
                    self.graphs[ch].add_node(node_name)

            # edges
            for comp in self.comparisons:
                for u1 in comp.matches[ch]['units1']:
                    u2 = comp.matches[ch]['hungarian_match_12'][u1]
                    if u2 != -1:
                        node1_name = str(comp.action1) + '_' + str(u1)
                        node2_name = str(comp.action2) + '_' + str(u2)
                        score = comp.matches[ch]['dissimilarity_scores'].loc[u1, u2]
                        self.graphs[ch].add_edge(node1_name, node2_name, weight=score)

            # the graph is symmetrical
            self.graphs[ch] = self.graphs[ch].to_undirected()

            # extract agrrement from graph
            if self._verbose:
                print('Multicomaprison step3: extract agreement from graph')

            self._new_units = {}
            added_nodes = []
            unit_id = 0

            # Note in this graph node=one unit for one sorter
            for node in self.graphs[ch].nodes():
                edges = self.graphs[ch].edges(node, data=True)
                session, unit = (str(node)).split('_')
                unit = int(unit)
                if len(edges) == 0:
                    avg_diss = 0
                    session_idxs = {session: unit}
                    self._new_units[unit_id] = {'avg_diss': 100,
                                                'session_unit_ids': session_idxs}
                    unit_id += 1
                    added_nodes.append(str(node))
                else:
                    # check if other nodes have edges (we should also check edges of
                    all_edges = list(edges)
                    for e in edges:
                        # Note for alessio n1>node1 n2>node2 e>edge
                        n1, n2, d = e
                        n2_edges = self.graphs[ch].edges(n2, data=True)
                        if len(n2_edges) > 0:  # useless line if
                            for e_n in n2_edges:
                                n_n1, n_n2, d = e_n
                                # Note for alessio  why do do you sorter each elements in the all_edges ?
                                if sorted([n_n1, n_n2]) not in [sorted([u, v]) for u, v, _ in all_edges]:
                                    all_edges.append(e_n)
                    avg_diss = np.mean([d['weight'] for u, v, d in all_edges])
                    min_edge = list(all_edges)[np.argmin([d['weight'] for u, v, d in all_edges])]

                    for edge in all_edges:
                        n1, n2, d = edge
                        if n1 not in added_nodes or n2 not in added_nodes:
                            session1, unit1 = n1.split('_')
                            session2, unit2 = n2.split('_')
                            unit1 = int(unit1)
                            unit2 = int(unit2)
                            session_idxs = {session1: unit1, session2: unit2}
                            if unit_id not in self._new_units.keys():
                                self._new_units[unit_id] = {'avg_diss': avg_diss,
                                                            'session_unit_ids': session_idxs}
                            else:
                                full_session_idxs = self._new_units[unit_id]['session_unit_ids']
                                for s, u in session_idxs.items():
                                    if s not in full_session_idxs:
                                        full_session_idxs[s] = u
                                self._new_units[unit_id] = {'avg_diss': avg_diss,
                                                            'session_unit_ids': full_session_idxs}
                            added_nodes.append(str(node))
                            if n1 not in added_nodes:
                                added_nodes.append(str(n1))
                            if n2 not in added_nodes:
                                added_nodes.append(str(n2))
                    unit_id += 1
            self.all_units[ch] = self._new_units


    def plot_matched_units(self, chan_group=None, ylim=[-200, 50], figsize=(10, 10)):
        '''

        Parameters
        ----------
        match_mode

        Returns
        -------

        '''
        if chan_group is None:
            ch_groups = self.all_units.keys()
        else:
            ch_groups = [chan_group]

        for ch_group in ch_groups:
            all_units = self.all_units[ch_group]
            all_matches = np.array([len(unit['session_unit_ids'].keys()) for unit in all_units.values()])

            if np.any(all_matches > 1):
                num_matches = len(np.where(all_matches > 1)[0])
                fig, ax_list = plt.subplots(nrows=len(self.action_list), ncols=num_matches, figsize=figsize)
                fig.suptitle('Channel group ' + str(ch_group))
                for i, ax in enumerate(ax_list):
                    ax[0].set_ylabel(self.action_list[i])

                id_ax = 0
                for u, unit in all_units.items():
                    sessions = list(unit['session_unit_ids'].keys())
                    unit_ids = list(unit['session_unit_ids'].values())
                    if len(sessions) > 1:
                        # retrieve templates
                        for comp in self.comparisons:
                            if comp.action1 in sessions:
                                action_idx = np.where(comp.action1 == np.array(self.action_list))[0][0]
                                session_idx = np.where(comp.action1 == np.array(sessions))[0][0]
                                unit_idx = np.where(comp.matches[ch_group]['units1'] ==
                                                    np.array(unit_ids)[session_idx])[0][0]
                                template = np.squeeze(comp.matches[ch_group]['templates1'][unit_idx])
                                ax_list[action_idx, id_ax].plot(template.T, color='C' + str(id_ax))
                                ax_list[action_idx, id_ax].set_title('Unit ' +
                                                                     str(comp.matches[ch_group]['units1'][unit_idx]))
                                ax_list[action_idx, id_ax].set_ylim(ylim)
                            if comp.action2 in sessions:
                                action_idx = np.where(comp.action2 == np.array(self.action_list))[0][0]
                                session_idx = np.where(comp.action2 == np.array(sessions))[0][0]
                                unit_idx = np.where(comp.matches[ch_group]['units2'] ==
                                                    np.array(unit_ids)[session_idx])[0][0]
                                template = np.squeeze(comp.matches[ch_group]['templates2'][unit_idx])
                                ax_list[action_idx, id_ax].plot(template.T, color='C' + str(id_ax))
                                ax_list[action_idx, id_ax].set_title('Unit ' +
                                                                     str(comp.matches[ch_group]['units2'][unit_idx]))
                                ax_list[action_idx, id_ax].set_ylim(ylim)

                        id_ax += 1

            else:
                print('Not matched units for group', ch_group)
                continue
