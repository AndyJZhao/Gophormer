import numba
import numpy as np
import scipy.sparse as sp
import torch as th
import dgl


@numba.njit(cache=True, locals={"_val": numba.float32, "res": numba.float32, "res_vnode": numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {inode: alpha}
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]: indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(
        adj_matrix.indptr, adj_matrix.indices, out_degree, numba.float32(alpha), numba.float32(epsilon), nodes, topk
    )

    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, normalization="row"):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk).tocsr()

    if normalization == "sym":
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1.0 / deg_sqrt

        row, col = topk_matrix.nonzero()
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == "col":
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1.0 / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == "row":
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return topk_matrix


def build_topk_ppr_matrix_from_data(edge_index, *args, **kwargs):
    if isinstance(edge_index, th.Tensor) or isinstance(edge_index, tuple):
        row, col = edge_index
        row, col = row.numpy(), col.numpy()
        n_node = int(max(row.max(), col.max())) + 1

        val = np.ones(row.shape[0])
        adj_matrix = sp.csr_matrix((val, (row, col)), shape=(n_node, n_node))
    else:
        adj_matrix = edge_index
    return topk_ppr_matrix(adj_matrix, *args, **kwargs)


def get_topk_ppr_matrix(g: dgl.DGLGraph):
    row, col = [_.numpy() for _ in g.edges()]
    adj_matrix = sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    idx = g.nodes().cpu().numpy()
    topk_matrix = topk_ppr_matrix(adj_matrix, idx=idx, alpha=0.5, eps=0.001, topk=32, normalization="row").tocoo()
    new_row, new_col, new_val = topk_matrix.row, topk_matrix.col, topk_matrix.data
    g_new = dgl.graph((new_row, new_col))
    g_new.edata['PPR'] = th.tensor(new_val).type(th.float32)
    return g_new
