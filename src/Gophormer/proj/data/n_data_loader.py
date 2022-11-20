from Gophormer.proj.pkg.dgl_utils import *
from Gophormer.functions.os_utils import *
from Gophormer.models.Gophormer import GophormerConfig
import torch as th
import os
from tqdm import tqdm
from Gophormer.proj.settings import *


class SampleNodeSubgraphLoader(th.utils.data.Dataset):  # Map style
    def __init__(self, g_sample, g_compute, cf: GophormerConfig, is_full, n_samples=None):
        # ! Load configs
        super(SampleNodeSubgraphLoader).__init__()
        self.__dict__.update(cf.__dict__)
        self.cf = cf
        self.sample_device, self.compute_dev = cf.sample_device, cf.compute_dev
        self.is_full = is_full
        self.n_samples = self.n_samples if n_samples is None else n_samples

        # ! Handling full-graph cases
        self.n_samples = self.n_samples if not self.is_full else 1
        self.max_samples = self.max_samples if not self.is_full else 1
        self.sample_f_prefix = f"{TEMP_PATH}{self.model}/sample_results/{self.dataset}/{self.dataset}_RmSL{int(self.rm_self_loop)}"
        if not self.is_full:
            self.sample_file = lambda _: f"{self.sample_f_prefix}_SP{self.sample}_{_}.sample"
        else:
            self.sample_file = lambda _: f"{self.sample_f_prefix}_FullInf.sample"

        # ! Initialization
        self.g_sample = g_sample.to(th.device('cuda:0'))
        self._init_sample()  # Graph with original adj, without edge features, used in sampling
        self.g_sample = self.g_sample.to(self.cf.sample_device)
        self.g_compute = g_compute.to(cf.compute_dev)  # With edge features, used in generation of subgraphs

    def _sample_and_save(self, node_id):
        resampled = False
        if os.path.exists(f_name := self.sample_file(node_id)):
            try:
                th.load(self.sample_file(node_id), map_location=self.sample_device)
                return  # Skip Sample if successfully loaded
            except:
                os.remove(f_name)  # Remove previous sample
                print(f'Node {node_id} failed to load.')
                resampled = True
        neighbors = [self._sample_node(node_id) for _ in range(self.max_samples)]
        if not os.path.exists(f_name):
            th.save(neighbors, f_name)
            if resampled:
                print(f'Node {node_id} resampled.')
        return

    def _sample_node(self, node_id):
        # ! GPU ACCESS FORBIDDEN IN THIS FUNC!! Otherwise workers must be 0
        induced_nodes = {0: node_id.view(-1)}
        cur_nodes = node_id
        for l, fanout in enumerate(self.fanouts):
            frontier = dgl.sampling.sample_neighbors(self.g_sample, cur_nodes, -1 if self.is_full else fanout)
            cur_nodes = frontier.edges()[0].unique()
            induced_nodes[l + 1] = cur_nodes
        sampled_nodes = th.cat(list(induced_nodes.values())).unique()
        return sampled_nodes

    def _init_sample(self):
        # ! Check if sampling is already completed
        if os.path.exists(self.sample_file(max(self.g_sample.nodes() - 1))):
            print('Already sampled')
        else:
            check_path(self.sample_f_prefix)
            for node_id in tqdm(self.g_sample.nodes(), desc='Sampling ego-graphs'):
                self._sample_and_save(node_id)
            print('Sampling finished.')
            from Gophormer.functions import exp_init
            exp_init(self.cf)
        return

    def __getitem__(self, node_id):
        """
        Return ego-subgraph nodes induced by seed nodes
        """

        def check_graph(graph_nodes):
            assert node_id in graph_nodes, 'Center node missing!'
            assert len(th.unique(graph_nodes)) == len(graph_nodes), 'Duplicate nodes founded!'

        selected_ids = np.random.choice(self.max_samples, self.n_samples, replace=False)
        try:
            neighbors = th.load(self.sample_file(node_id), map_location=self.sample_device)
            [check_graph(neighbors[_]) for _ in selected_ids]
        except:
            neighbors = [self._sample_node(node_id) for _ in range(self.max_samples)]
            th.save(neighbors, self.sample_file(node_id))

        return node_id, [neighbors[_] for _ in selected_ids]
        # def _construct_subgraph(sampled_nodes):
        #     # Add fully connected edges
        #     sub_g = dgl.node_subgraph(self.g_compute, sampled_nodes)
        #     sub_g = to_fully_connected_graph(sub_g, self.compute_dev)
        #     return sub_g
        # selected_ids = np.random.choice(self.max_samples, self.n_samples, replace=False)
        # sub_g_list = [_construct_subgraph(neighbors[_]) for _ in selected_ids]
        # return node_id, sub_g_list

    def get_batch_data(self, batch_data):
        """

        Args:
            batch_data: node_ids and neighbor list (induced nodes)

        Returns:
            node_ids: Batch of center node-id (id at the entire graph).
            center_node_pos: Batch of center node-pos (id at the batched graph).
            batched_graphs: DGL batched ego-graphs induced by center nodes.
            graph_offset: Starting point of batched graphs.

        """

        def _construct_subgraph(sampled_nodes):
            # Add fully connected edges
            sub_g = dgl.node_subgraph(self.g_compute, sampled_nodes.to(self.compute_dev))
            sub_g = to_fully_connected_graph(sub_g, self.compute_dev)
            return sub_g

        _node_ids, _neighbor_list = batch_data
        sample_ids = range(self.n_samples)
        node_ids = th.stack([id for id in _node_ids for _ in sample_ids])  # Repeat node ids to [bsz * n_samples]
        graph_list = [_construct_subgraph(neighbors[_]) for neighbors in _neighbor_list for _ in sample_ids]
        # bsz, bsz * n_samples

        batched_graphs = dgl.batch(graph_list)

        # ! Calculate graph offset
        g_sizes = batched_graphs.batch_num_nodes()
        graph_offset = th.zeros_like(g_sizes)
        if len(g_sizes) > 1:
            graph_offset[range(1, len(g_sizes))] = th.stack([g_sizes[:_].sum() for _ in range(1, len(g_sizes))]).squeeze()
        # ! Calculate node positions
        id_lookup = lambda g, id: th.where(g.ndata['_ID'] == id)[0]
        try:
            center_node_pos = th.stack([id_lookup(g, node_ids[id]) + graph_offset[id]
                                        for id, g in enumerate(graph_list)]).squeeze()
        except:
            print('NodePos lookup ERROR!!! Nodes in graph:')
            print([g.nodes() for id, g in enumerate(graph_list)])

        return node_ids, center_node_pos, batched_graphs, graph_offset
