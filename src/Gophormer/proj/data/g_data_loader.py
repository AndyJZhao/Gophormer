from Gophormer.proj.pkg.dgl_utils import *
from Gophormer.functions.os_utils import *
from Gophormer.models.Gophormer import GophormerConfig
import torch as th
import os
from tqdm import tqdm
from Gophormer.proj.settings import *

# BAD_SEED_NODES = [47084]
BAD_SEED_NODES = []


# BAD_SEED_NODES = [12040, 168102, 110268, 118535, 24791, 18697, ]  # Arxiv
# BAD_SEED_NODES = [132677, 12040, 18697, 24791, 118535, 110268, 12040]  # Arxiv


class SampleGraphSubgraphLoader(th.utils.data.Dataset):  # Map style
    def __init__(self, g_sample, g_compute, cf: GophormerConfig, is_full, n_samples=None):
        # ! Load configs
        super(SampleGraphSubgraphLoader).__init__()
        self.__dict__.update(cf.__dict__)
        self.sample_device, self.compute_dev = cf.sample_device, cf.compute_dev
        self.is_full = is_full
        self.n_samples = self.n_samples if n_samples is None else n_samples
        self.log = cf.logger.log

        # ! Handling full-graph cases
        self.n_samples = self.n_samples if not self.is_full else 1
        self.max_samples = self.max_samples if not self.is_full else 1
        # cf.sample_f_prefix = f"{TEMP_PATH}{self.model}/sample_results/{self.dataset}/{self.dataset}_RmSL{int(self.rm_self_loop)}"
        if not self.is_full:
            self.sample_file = lambda _: f"{cf.sample_f_prefix}_SP{self.sample}_{_}.sample"
        else:
            self.sample_file = lambda _: f"{cf.sample_f_prefix}_FullInf.sample"

        # ! Initialization
        self.g_sample = g_sample.to(cf.sample_device)
        self.g_compute = g_compute.to(cf.compute_dev)  # With edge features, used in generation of subgraphs
        check_path(cf.sample_f_prefix)
        self.re_sample_cnt = 0

    def _node2seq(self, node_id):
        def _construct_subgraph(sampled_nodes):
            # Add fully connected edges
            sub_g = dgl.node_subgraph(self.g_compute, sampled_nodes.to(self.compute_dev))
            sub_g = to_fully_connected_graph(sub_g, self.compute_dev)
            return sub_g

        random_samples = np.random.choice(self.max_samples, self.n_samples, replace=False)
        try:
            if self.re_sample:
                raise KeyError('Resample')
            sub_g_list = dgl.load_graphs(self.sample_file(node_id), random_samples.tolist())[0]
            # assert th.sum(self.g_compute.ndata['F'][node_id] != batched_graphs.ndata['F'][node_pos]) == 0  # Center node
        except:
            neighbors = [self._sample_single(node_id) for _ in range(self.max_samples)]
            sub_g_list = [_construct_subgraph(neighbors[_]).to(self.sample_device) for _ in range(self.max_samples)]
            dgl.save_graphs(self.sample_file(node_id), sub_g_list)
            sub_g_list = [sub_g_list[_] for _ in random_samples]
        return sub_g_list

    def _sample_single(self, node_id):
        induced_nodes = {0: node_id.view(-1)}
        cur_nodes = node_id
        for l, fanout in enumerate(self.fanouts):
            frontier = dgl.sampling.sample_neighbors(self.g_sample, cur_nodes, -1 if self.is_full else fanout)
            cur_nodes = frontier.edges()[0].unique()
            induced_nodes[l + 1] = cur_nodes
        sampled_nodes = th.cat(list(induced_nodes.values())).unique()
        return sampled_nodes

    def __getitem__(self, node_id):
        """
        Return ego-subgraph nodes induced by seed nodes
        """
        return node_id

    def get_batch_data(self, _node_ids, rm_prev=False):
        """

        Args:
            _node_ids: node_ids and neighbor list (induced nodes)

        Returns:
            node_ids: Batch of center node-id (id at the entire graph).
            node_pos: Batch of center node-pos (id at the batched graph).
            batched_graphs: DGL batched ego-graphs induced by center nodes.
            graph_offset: Starting point of batched graphs.

        """

        def rm_ego_graphs():
            file_list = [self.sample_file(id) for id in node_ids]
            self.log(f'Erroneous batch starting with {node_ids[0]}, sampled egographs will be removed.', 'ERROR')
            [silent_remove(_) for _ in file_list]

        # self.log(f'Start batching {_node_ids}')
        node_ids = th.stack([id for id in _node_ids for _ in range(self.n_samples)])  # Repeat node ids to [bsz * n_samples]
        for _ in BAD_SEED_NODES:
            if _ in node_ids.cpu().numpy().tolist():
                self.log('re_sample the batch afterwards:')
                self.re_sample_cnt = 2
        if rm_prev or self.re_sample_cnt > 0:
            self.log(f'Removed batch data {_node_ids}')
            rm_ego_graphs()
            self.re_sample_cnt -= 1

        graph_list = [_ for id in _node_ids for _ in self._node2seq(id)]
        batched_graphs = dgl.batch(graph_list)

        # ! Calculate graph offset
        g_sizes = batched_graphs.batch_num_nodes()
        graph_offset = th.zeros_like(g_sizes)
        if len(g_sizes) > 1:
            graph_offset[range(1, len(g_sizes))] = th.stack([g_sizes[:_].sum() for _ in range(1, len(g_sizes))]).squeeze()
        # ! Calculate node positions
        id_lookup = lambda g, id: th.where(g.ndata['_ID'] == id)[0]
        node_pos = th.stack([id_lookup(g, node_ids[id]) + graph_offset[id]
                             for id, g in enumerate(graph_list)]).squeeze()

        node_ids, node_pos, batched_graphs, graph_offset = [_.to(self.compute_dev) for _ in (node_ids, node_pos, batched_graphs, graph_offset)]
        try:
            # print('Asserting')
            assert th.sum(self.g_compute.ndata['F'][node_ids] != batched_graphs.ndata['F'][node_pos]) == 0  # Center node
        except:
            rm_ego_graphs()
            self.log(f'resampled {_node_ids}')
            return self.get_batch_data(node_ids.to(self.sample_device))
        self.log(f'Finished batch {_node_ids[0]}', 'DEBUG')
        return node_ids, node_pos, batched_graphs, graph_offset
