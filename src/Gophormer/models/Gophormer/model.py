import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from Gophormer.models.Gophormer.layers import GophormerLayer
from Gophormer.models.Gophormer.layers import MLPReadout, OrderPoolingReadout, OrderPooling
from Gophormer.functions.util_funcs import lot_to_tol, init_random_state


class Gophormer(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.__dict__.update(cf.__dict__)
        self.in_feat_dropout = nn.Dropout(cf.in_feat_dropout)
        self.readout_fn = cf.readout.split('_')[0]
        self.readout_n_layer = int(cf.readout.split('_')[1])
        if self.feat_map_layer:
            self.feat_map_layer = nn.Linear(cf.feat_dim, cf.gt_hidden_dim)
            self.layers = nn.ModuleList(
                [GophormerLayer(cf.gt_hidden_dim, cf.gt_hidden_dim, cf)
                 for _ in range(self.gt_n_layers - 1)])
        else:
            self.layers = nn.ModuleList(
                [GophormerLayer(cf.feat_dim, cf.gt_hidden_dim, cf)])
            for _ in range(self.gt_n_layers - 2):
                self.layers.append(
                    GophormerLayer(cf.gt_hidden_dim, cf.gt_hidden_dim, cf))

        if self.readout_n_layer > 0:
            self.layers.append(
                GophormerLayer(cf.gt_hidden_dim, cf.gt_hidden_dim, cf))
            if self.readout_fn == 'MLP':
                self.readout_pos = cf.readout.split('_')[2]
                self.readout_layer = MLPReadout(cf.gt_hidden_dim, cf.n_class, self.readout_n_layer, cf.readout_dropout)
            elif self.readout_fn in ['OrderPooling', 'SPD']:
                readout_pooling = cf.readout.split('_')[2]
                self.readout_layer = OrderPoolingReadout(cf.gt_hidden_dim, cf.n_class, self.depth, self.readout_n_layer, readout_pooling, cf.readout_dropout)
        else:
            self.layers.append(
                GophormerLayer(cf.gt_hidden_dim, cf.n_class, cf, last_layer=True))
            if self.readout_fn in ['OrderPooling', 'SPD']:
                readout_pooling = cf.readout.split('_')[2]
                self.order_pooling = OrderPooling(cf.n_class, self.depth, readout_pooling)

        if self.global_nodes > 0:
            self.global_context = nn.Embedding(self.global_nodes, cf.gt_hidden_dim)

        # Print and init random variable
        cf.logger.log(f'Model Config:{cf}\n{self}')
        init_random_state(cf.seed)

    def forward(self, g, center_nodes, graph_offset):
        '''

        Args:
            g: dgl batched graph
            pos_info: relative position to subgraphs

        Returns:

        '''
        h = g.ndata['F'].to(self.compute_dev)
        h = self.in_feat_dropout(h)
        if self.feat_map_layer:
            h = F.dropout(self.feat_map_layer(h), self.dropout)
            h = F.relu(h)
            # Add global nodes + CE before Gophormer-layer
            g, h = self.process_graph(g, h, graph_offset)

        for l, conv in enumerate(self.layers):
            h = conv(g, h, is_first_layer=l == 0)
            if l == 0 and not self.feat_map_layer:  # Add global nodes + CE
                g, h = self.process_graph(g, h, graph_offset)

        if self.readout_n_layer > 0:
            if self.readout_fn == 'MLP':
                if self.readout_pos == 'center':
                    h_out = self.readout_layer(h[center_nodes])
            else:
                return NotImplementedError
        else:
            if self.readout_fn == 'MLP':
                h_out = h[center_nodes]
            else:
                return NotImplementedError

        return h_out

    def process_graph(self, g, h, graph_offset):
        '''
        Add global nodes and use centrality encoding
        '''
        if self.global_nodes > 0:
            n_graphs = len(graph_offset)
            n_nodes = g.num_nodes()  # Total nodes in batched graph
            subg_node_range = [(int(graph_offset[_]), int(graph_offset[_ + 1]))
                               for _ in range(n_graphs - 1)]  #
            subg_node_range.append((graph_offset[-1], n_nodes))
            gn_id = lambda g_id, u: n_nodes + g_id * self.global_nodes + u
            edges_to_add = sum([[(gn_id(g_id, gn), rn), (rn, gn_id(g_id, gn))]
                                for g_id, (start, end) in enumerate(subg_node_range)
                                for rn in range(start, end)  # each real node in subgraph
                                for gn in range(self.global_nodes)  # each global node
                                ], [])  # Concat edge tuple list to one list
            uids, vids = lot_to_tol(edges_to_add)
            # Prepare node and edge features
            # FIXME Edges between GNs

            n_feat = self.global_context(th.tensor([_ for __ in range(n_graphs)
                                                    for _ in range(self.global_nodes)]).to(g.device))
            g = dgl.add_nodes(g, n_graphs * self.global_nodes)

            if self.order_bias:
                e_feat = th.zeros((len(uids), g.edata['ProximityEncoding'].shape[1])).to(g.device)
                e_feat[:, -1] = 1
                g = dgl.add_edges(g, uids, vids, data={'ProximityEncoding': e_feat})
            h = th.cat((h, n_feat))
        return g, h
