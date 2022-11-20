from itertools import product
from time import time

import dgl
import numpy as np
import torch as th


def get_edge_set(g: dgl.DGLGraph):
    '''graph_edge_to list of (row_id, col_id) tuple
    '''

    return set(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))


def to_fully_connected_graph(g: dgl.DGLGraph, device):
    """
    Args:
        g: DGLGraph to be converted
    Returns: converted fully connected graph
    """
    print_log = True
    print_log = False
    implementation = 'Add Non-exist by itertools and set()'

    def tik(t_list, info='', print_log=print_log):
        cur = time()
        if print_log:
            print(f'Implementation={implementation}.{info} time {cur - t_list[-1]}')
        return t_list + [cur]

    t_list = [time()]

    if implementation == 'Add Non-exist by itertools and set()':
        # Add non-existing edges to graph, slightly Faster
        full_edge_set = set(product(g.cpu().nodes().numpy(), repeat=2))
        existing_edges = get_edge_set(g)
        edges_to_add = full_edge_set - existing_edges
        if len(edges_to_add) > 0:
            row_inds, col_inds = edge_set_to_inds(edges_to_add)
            return dgl.add_edges(g, row_inds, col_inds)
        else:
            return g
    elif implementation == 'Add None-exist by th.unique':
        full_edges = th.cartesian_prod(g.nodes(), g.nodes())
        ori_edges = th.column_stack((g.edges()[0], g.edges()[1]))
        edges, counts = th.cat((full_edges, ori_edges)).unique(dim=0, return_counts=True)
        edges_to_add = edges[counts == 1]
        return dgl.add_edges(g, edges_to_add[0], edges_to_add[1])
    elif implementation == 'Add FC to existing, remove duplicate':
        # Add existing edges and remove duplicate (by to_simple function)
        full_edges = th.cartesian_prod(g.nodes(), g.nodes())
        g_new = g.add_edges(full_edges[:, 0], full_edges[:, 1])
        return dgl.to_simple(g_new.cpu(), copy_edata=True).to(g.device)
    elif implementation == 'New FC graph':
        # Num add/remove operations: 2x ori_subg n_edges
        full_edges = th.cartesian_prod(g.nodes(), g.nodes())
        g_new = dgl.graph((full_edges[:, 0], full_edges[:, 1]), device=device)
        # Remove original edges with no edge feature
        g_new.remove_edges(g_new.edge_ids(g.edges()[0], g.edges()[1]))
        # Add new edges with edge feature
        g_new.add_edges(*g.edges(), data=g.edata)
        g_new.ndata['F'] = g.ndata['F']
        return g_new
