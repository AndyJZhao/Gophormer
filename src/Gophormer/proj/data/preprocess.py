import pickle
from collections import Counter
import os

import dgl
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import train_test_split

import Gophormer.functions as uf
from Gophormer.proj.settings import *
from types import SimpleNamespace


def edge_lists_to_set(_):
    return set(list(map(tuple, _)))


def stratified_train_test_split(label_idx, labels, n_nodes, train_rate, dataset=''):
    if dataset == 'cora':
        seed = 0
    else:
        seed = 2021
    n_train_nodes = int(train_rate / 100 * n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - n_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate_in_labeled_nodes, random_state=seed, shuffle=True, stratify=labels)
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=.5, random_state=seed, shuffle=True, stratify=labels[test_and_valid_idx])
    return train_idx, valid_idx, test_idx


def row_normalization(mat):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mat.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat


def normalize_sparse_tensor(adj):  # from DeepRobust
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    n_nodes = adj.size(0)

    row, col = edge_index
    import torch  # import torch first to avoid libc10_cuda.so error
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    shape = adj.shape
    return th.sparse.FloatTensor(edge_index, values, shape)


def preprocess_data(dataset, train_percentage):
    # Modified from AAAI21 FA-GCN
    if dataset in ['cora', 'citeseer', 'pubmed']:
        load_default_split = train_percentage <= 0
        edge = np.loadtxt(f'{DATA_PATH}{dataset}/{dataset}.edge', dtype=int).tolist()
        features = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.feature')
        labels = np.loadtxt(f'{DATA_PATH}{dataset}/{dataset}.label', dtype=int)
        if load_default_split:
            train = np.loadtxt(f'{DATA_PATH}{dataset}/{dataset}.train', dtype=int)
            val = np.loadtxt(f'{DATA_PATH}{dataset}/{dataset}.val', dtype=int)
            test = np.loadtxt(f'{DATA_PATH}{dataset}/{dataset}.test', dtype=int)
        else:
            train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels), train_percentage, dataset)
        nclass = len(set(labels.tolist()))

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        features = row_normalization(features)
        features = th.FloatTensor(features)
        labels = th.LongTensor(labels)
        train = th.LongTensor(train)
        val = th.LongTensor(val)
        test = th.LongTensor(test)

    elif dataset in ['airport', 'blogcatalog', 'flickr']:
        load_default_split = train_percentage <= 0
        adj_orig = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_adj.pkl', 'rb'))  # sparse
        features = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_features.pkl', 'rb'))  # sparase
        labels = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_labels.pkl', 'rb'))  # tensor
        if th.is_tensor(labels):
            labels = labels.numpy()

        if load_default_split:
            tvt_nids = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_tvt_nids.pkl', 'rb'))  # 3 array
            train = tvt_nids[0]
            val = tvt_nids[1]
            test = tvt_nids[2]
        else:
            train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels),
                                                           train_percentage)
        nclass = len(set(labels.tolist()))

        adj_orig = adj_orig.tocoo()
        U = adj_orig.row.tolist()
        V = adj_orig.col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        if dataset in ['airport']:
            features = row_normalization(features)

        if sp.issparse(features):
            features = th.FloatTensor(features.toarray())
        else:
            features = th.FloatTensor(features)

        labels = th.LongTensor(labels)
        train = th.LongTensor(train)
        val = th.LongTensor(val)
        test = th.LongTensor(test)

    elif dataset in ['dblp']:
        fname = f'{DATA_PATH}dblp/processed_dblp.pickle'
        if os.path.exists(fname):
            from torch_geometric.datasets import CitationFull
            import torch_geometric.transforms as T
            data = CitationFull(root=f'{DATA_PATH}', name=dataset, transform=T.NormalizeFeatures())[0]
            edges = data.edge_index
            features = data.x.numpy()
            labels = data.y.numpy()
            data_dict = {'edges': edges, 'features': features, 'labels': labels}
            uf.save_pickle(data_dict, fname)
        else:
            data_dict = uf.load_pickle(fname)
        edges, features, labels = data_dict['edges'], data_dict['features'], data_dict['labels']
        train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels), train_percentage)
        nclass = len(set(labels.tolist()))

        U = edges[0]
        V = edges[1]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        features = th.FloatTensor(features)
        labels = th.LongTensor(labels)
        train = th.LongTensor(train)
        val = th.LongTensor(val)
        test = th.LongTensor(test)
    elif dataset in ['arxiv']:
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-arxiv',
                                         root=uf.check_path('./data/ogb_raw/ogb_arxiv'))
        split_idx = dataset.get_idx_split()
        train, val, test = split_idx["train"], split_idx["valid"], split_idx["test"]
        g, labels = dataset[0]
        features = g.ndata['feat']
        nclass = 40
        labels = labels.squeeze()
        g = dgl.to_bidirected(g)

    g = dgl.add_self_loop(g)  # ? Add or not?? Potentially buggy if removed.
    supervision = SimpleNamespace(train_x=train, val_x=val, test_x=test, labels=labels)

    return g, features, features.shape[1], nclass, supervision


@uf.time_logger
def proximity_encoding(g: dgl.DGLGraph, K, include_identity, rm_self_loop, global_nodes, pe_file):
    '''
    Encode the proximity unde K-order into the edge feature of a graph.
    Args:
        g: a DGLGraph.
        K: maximum number of order
        include_identity: whether A^0 (Identity matrix) is included.
        rm_self_loop: whether the base matrix is A or A+I
        global_nodes: global_nodes

    Returns: g_compute: the graph to compute on with edge features of proximity encoding

    '''

    def get_edges_and_weights(adj_mat):
        def _check_redundant(row_inds, col_inds):
            edge_list = list(zip(row_inds.cpu().numpy(), col_inds.cpu().numpy()))
            rendundant_edges = [item for item, count in Counter(edge_list).items() if count > 1]
            assert len(rendundant_edges) == 0, 'Redundant edges found'

        normed_adj = normalize_sparse_tensor(adj_mat)
        row_inds, col_inds = normed_adj._indices()
        edge_set = set(list(zip(row_inds.cpu().numpy(), col_inds.cpu().numpy())))
        _check_redundant(row_inds, col_inds)
        return edge_set, (row_inds, col_inds), normed_adj._values()

    if os.path.exists(pe_file):
        print('Loaded previous Proximity Encoding')
        g_compute, _ = dgl.load_graphs(pe_file)
        g_compute = g_compute[0]
    else:
        A = dgl.remove_self_loop(g).adj() if rm_self_loop else g.adj()
        cur_edges_set, edge_inds, edge_weights = set(), {}, {}

        # ! Step 1: Calculate proximity using sparse matrix
        # Calculate A^0, A^1, ..., A^K recurrently
        for k in range(0 if include_identity else 1, K + 1):
            # Compute A^k
            if k == 0:
                cur_adj = th_sparse_identity_mat(g.num_nodes())
            else:
                cur_adj = th.sparse.mm(cur_adj, A) if k > 1 else A.clone()
            # Get edges in A^k and normalized edge weights
            edge_set_k, edge_inds[k], edge_weights[k] = get_edges_and_weights(cur_adj)
            # Update edges
            cur_edges_set |= edge_set_k

        # ! Step 2: Build a new graph with edge feature as proximities
        new_row, new_col = map(list, zip(*list(cur_edges_set)))
        g_compute = dgl.graph((new_row, new_col))
        weights = []
        for k in range(0 if include_identity else 1, K + 1):
            # weights = weights_at_k_order else 0]
            weights.append(th.zeros((g.num_nodes(), g.num_nodes())))
            weights[-1][edge_inds[k]] = edge_weights[k]
            # g_new.edata[f'ProximityEncoding{k}'] = weights[edge_inds[K]]
        if global_nodes > 0:
            weights.append(th.zeros((g.num_nodes(), g.num_nodes())))
        weights = th.stack(weights, dim=2)  # weights~ [N, N, num_relations]
        g_compute.edata['ProximityEncoding'] = weights[g_compute.edges()]

        # ! Step 3: Corner case handling
        # If proximity encoding is identity only, i.e. cf.proximity_encoding='0i', the original edges in the generated graph and should be added.
        if K == 0:  # Add non-self-loop edges for K=0
            g_wo_self_loop = dgl.remove_self_loop(g).to(g_compute.device)
            g_compute = dgl.add_edges(g_compute, g_wo_self_loop.edges()[0], g_wo_self_loop.edges()[1])
        if not os.path.exists(pe_file):
            dgl.save_graphs(pe_file, [g_compute])

    return g_compute


def th_sparse_identity_mat(N):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = th.arange(N).long().unsqueeze(0).expand(2, N)
    values = th.tensor(1.0).expand(N)
    return th.sparse_coo_tensor(indices, values, (N, N))
