import numpy as np
import networkx as nx
from .trackunitcomparison import TrackingSession
from .data_processing import get_data_path, get_channel_groups, load_spiketrains
from .track_units_tools import get_unit_id, compute_templates, plot_waveform, lighten_color, plot_template
import matplotlib.pylab as plt
from tqdm import tqdm
import uuid
from matplotlib import gridspec
from collections import defaultdict
from pathlib import Path

class TrackMultipleSessions:
    def __init__(self, action_list, actions, channel_group=None,
                 max_dissimilarity=None,
                 verbose=False, progress_bar=None, data_path=None):
        self.data_path = Path.cwd() if data_path is None else Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.action_list = action_list
        self._actions = actions
        self._channel_group = channel_group
        self.max_dissimilarity = max_dissimilarity or np.inf
        self._verbose = verbose
        self._pbar = tqdm if progress_bar is None else progress_bar

        if self._channel_group is None:
            dp = get_data_path(self._actions[self.action_list[0]])
            self._channel_groups = get_channel_groups(dp)
        else:
            self._channel_groups = [self._channel_group]

        self._do_matching()
        self._make_graph()
        self._identify_units()

    def _do_matching(self):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step1: pairwise comparison')

        self.comparisons = []
        N = len(self.action_list)
        pbar = self._pbar(total=int((N**2 - N) / 2))
        for i in range(N):
            for j in range(i + 1, N):
                if self._verbose:
                    print("  Comparing: ", self.action_list[i], " and ", self.action_list[j])
                comp = TrackingSession(
                    self.action_list[i], self.action_list[j],
                    actions=self._actions,
                    max_dissimilarity=self.max_dissimilarity,
                    channel_group=self._channel_group,
                    verbose=self._verbose)
                # comp.save_dissimilarity_matrix()
                self.comparisons.append(comp)
                pbar.update(1)
        pbar.close()

    def _make_graph(self, max_dissimilarity=None):
        max_dissimilarity = max_dissimilarity or self.max_dissimilarity
        if self._verbose:
            print('Multicomaprison step2: make graph')

        self.graphs = {}

        for ch in self._channel_groups:
            if self._verbose:
                print('Processing channel', ch)
            self.graphs[ch] = nx.Graph()

            # nodes
            for comp in self.comparisons:
                # if same node is added twice it's only created once
                for i, action_id in enumerate(comp.action_ids):
                    for u in comp.unit_ids[ch][i]:
                        node_name = action_id + '_' + str(u)
                        self.graphs[ch].add_node(
                            node_name, action_id=action_id,
                            unit_id=u)

            # edges
            for comp in self.comparisons:
                if 'hungarian_match_01' not in comp.matches[ch]:
                    continue
                for u1 in comp.unit_ids[ch][0]:
                    u2 = comp.matches[ch]['hungarian_match_01'][u1]
                    if u2 != -1:
                        score = comp.matches[ch]['dissimilarity_scores'].loc[u1, u2]
                        if score <= max_dissimilarity:
                            node1_name = comp.action_id_0 + '_' + str(u1)
                            node2_name = comp.action_id_1 + '_' + str(u2)
                            self.graphs[ch].add_edge(node1_name, node2_name, weight=score)

            # the graph is symmetrical
            self.graphs[ch] = self.graphs[ch].to_undirected()

    def save_graphs(self):
        for ch, graph in self.graphs.items():
            nx.readwrite.write_yaml(graph, self.data_path / f'graph-group-{ch}.yaml')

    def _identify_units(self):
       if self._verbose:
           print('Multicomaprison step3: extract agreement from graph')
       self.identified_units = {}
       for ch in self._channel_groups:
            # extract agrrement from graph
            self._new_units = {}
            graph = self.graphs[ch]
            for node_set in nx.connected_components(graph):
                unit_id = str(uuid.uuid4())
                edges = graph.edges(node_set, data=True)

                if len(edges) == 0:
                    average_dissimilarity = None
                else:
                    average_dissimilarity = np.mean(
                        [d['weight'] for _, _, d in edges])

                original_ids = defaultdict(list)
                for node in node_set:
                    original_ids[graph.nodes[node]['action_id']].append(
                        graph.nodes[node]['unit_id']
                    )

                self._new_units[unit_id] = {
                    'average_dissimilarity': average_dissimilarity,
                    'original_unit_ids': original_ids}

            self.identified_units[ch] = self._new_units

    def _get_waveforms(self, action_id, unit_id, channel_group):
        for comp in self.comparisons:
            if action_id in comp.action_ids:
                i = comp.action_ids.index(action_id)
                unit_ids = list(comp.unit_ids[channel_group][i])
                unit_idx = unit_ids.index(unit_id)
                wf = getattr(comp, f'waveforms_{i}')(channel_group)[unit_idx]
                break
        return wf

    def _get_template(self, action_id, unit_id, channel_group):
        for comp in self.comparisons:
            if action_id in comp.action_ids:
                i = comp.action_ids.index(action_id)
                unit_ids = list(comp.unit_ids[channel_group][i])
                unit_idx = unit_ids.index(unit_id)
                template = comp.templates[channel_group][i][unit_idx]
                break
        return template

    def redo_match(self, max_dissimilarity):
        self.max_dissimilarity = max_dissimilarity
        self._make_graph(max_dissimilarity=max_dissimilarity)
        self._identify_units()

    def plot_matches(self, style='template', chan_group=None, figsize=(10, 3)):
        '''

        Parameters
        ----------
        style: 'template' or 'waveform'

        Returns
        -------

        '''
        if chan_group is None:
            ch_groups = self.identified_units.keys()
        else:
            ch_groups = [chan_group]

        for ch_group in ch_groups:
            identified_units = self.identified_units[ch_group]
            units = [
                unit['original_unit_ids']
                for unit in identified_units.values()
                if len(unit['original_unit_ids']) > 1]
            num_units = sum([len(u) for u in units])
            fig = plt.figure(figsize=(figsize[0], figsize[1] * num_units))
            gs = gridspec.GridSpec(num_units, 1)
            fig.suptitle('Channel group ' + str(ch_group))
            id_ax = 0
            for unit in units:
                axs = None
                for action_id, unit_ids in unit.items():
                    for unit_id in unit_ids:

                        label = action_id + ' Unit ' + str(unit_id)
                        if axs is None:
                            color = 'C' + str(id_ax)

                        if style == 'waveform':
                            waveforms = self._get_waveforms(
                                action_id, unit_id, ch_group)
                            axs = plot_waveform(
                                waveforms,
                                fig=fig, gs=gs[id_ax], axs=axs,
                                color=color, label=label)
                        elif style == 'template':
                            template = self._get_template(action_id, unit_id, ch_group)
                            axs = plot_template(
                                template,
                                fig=fig, gs=gs[id_ax], axs=axs,
                                color=color, label=label)
                        else:
                            raise ValueError('style must be "template" or "waveform"')
                        color = lighten_color(color)
                id_ax += 1
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
