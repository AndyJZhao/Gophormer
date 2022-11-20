import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def biased_exp(field_in, field_out, bias_field=None):
    def func(edges):
        if bias_field is None:
            return {field_out: th.exp((edges.data[field_in].sum(-1, keepdim=True)).clamp(-5, 5))}
        else:
            return {field_out: th.exp((edges.data[field_in].sum(-1, keepdim=True) + edges.data[bias_field]).clamp(-5, 5))}

    return func


def calc_att_bias(in_field, out_field, Linear, num_heads):
    def func(edges):
        return {out_field: Linear(edges.data[in_field]).unsqueeze(-1).repeat(1, num_heads, 1)}  # Ex8x1

    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, bias, order_bias=None, head_mode='h0', K=1, include_identity=True, last_layer=False, global_nodes=0):
        super().__init__()

        self.out_dim = out_dim
        self.n_heads = n_heads
        self.order_bias = order_bias
        self.head_mode = head_mode
        self.last_layer = last_layer

        self.Q = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.K = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.V = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        if order_bias:
            num_relations = K + 1 if include_identity else K
            num_relations = num_relations + 1 if global_nodes > 0 else num_relations
            if self.head_mode == 'h0':
                self.W = nn.Linear(num_relations, 1, bias=bias)  # A^0 (Identity) ~ A^K
            else:
                self.W = nn.Linear(num_relations, n_heads, bias=bias)

            # Initialize bias to zero
            th.nn.init.constant_(self.W.weight, 0)

    def propagate_attention(self, g):
        # Q x K
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # ? Use available edge features to modify the scores for edges
        if self.order_bias:
            if self.head_mode == 'h0':
                g.apply_edges(calc_att_bias('ProximityEncoding', 'att_bias', self.W, self.n_heads))
            else:
                g.apply_edges(calc_att_bias('ProximityEncoding', 'att_bias', self.W, 1))

            g.apply_edges(biased_exp('score', 'score_soft', 'att_bias'))
        else:
            g.apply_edges(biased_exp('score', 'score_soft'))

        # Send weighted values to target nodes
        eids = g.edges()
        if self.last_layer:
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'final_wV'))
            g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'final_z'))
        else:
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))

    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [n_nodes, n_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.n_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.n_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.n_heads, self.out_dim)

        self.propagate_attention(g)

        if self.last_layer:
            h_out = g.ndata['final_wV'] / (g.ndata['final_z'] + th.full_like(g.ndata['final_z'], 1e-6))
        else:
            h_out = g.ndata['wV'] / (g.ndata['z'] + th.full_like(g.ndata['z'], 1e-6))

        return h_out


class GophormerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, cf, last_layer=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.__dict__.update(cf.__dict__)
        self.last_layer = last_layer
        self.head_mode, self.layer_mode = self.bias_mode.split('_')

        if self.last_layer:
            self.attention = MultiHeadAttentionLayer(in_dim, out_dim, 1, self.bias, self.order_bias, self.head_mode, self.K, self.include_identity, last_layer=self.last_layer,
                                                     global_nodes=self.global_nodes)
            self.O_h = nn.Linear(out_dim, out_dim)

            if self.norm[-2:] == 'LN':
                self.layer_norm_input_feat = nn.LayerNorm(in_dim)
                self.layer_norm1_h = nn.LayerNorm(out_dim)

        else:
            self.attention = MultiHeadAttentionLayer(in_dim, out_dim // self.gt_n_heads, self.gt_n_heads, self.bias, self.order_bias, self.head_mode, self.K, self.include_identity,
                                                     global_nodes=self.global_nodes)

            self.O_h = nn.Linear(out_dim, out_dim)

            if self.norm[-2:] == 'LN':
                self.layer_norm_input_feat = nn.LayerNorm(in_dim)
                self.layer_norm1_h = nn.LayerNorm(out_dim)

            # FFN for h
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
            self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

            if self.norm[-2:] == 'LN':
                self.layer_norm2_h = nn.LayerNorm(out_dim)

    def forward(self, g, h, is_first_layer):
        h_in1 = h  # for first residual connection
        if self.norm == 'PreLN':  # PreLayerNorm
            if is_first_layer:
                h = self.layer_norm_input_feat(h)
            else:
                h = self.layer_norm1_h(h)

        # multi-head attention out
        h_attn_out = self.attention(g, h)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_dim)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)

        if not self.last_layer:
            # residual connection
            # The first residual connection cannot be performed since the feat_dim doesn't match the gt_hidden_dim.
            if not is_first_layer:
                h = h_in1 + h

            if self.norm == 'LN':  # PostLayerNorm
                h = self.layer_norm1_h(h)
            h_in2 = h  # for second residual connection
            if self.norm == 'PreLN':
                h = self.layer_norm2_h(h)

            # FFN for h
            h = self.FFN_h_layer1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.FFN_h_layer2(h)

            # residual connection
            h = h_in2 + h
            if self.norm == 'LN':
                h = self.layer_norm2_h(h)
        return h


class MLPReadout(nn.Module):
    """
        MLP Layer used after graph vector representation
        Dropout is added compared with original implementation of SAN
    """

    def __init__(self, input_dim, output_dim, L=2, dropout=0.5):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.drop(self.FC_layers[l](y))
            y = F.relu(y)
        y = self.drop(self.FC_layers[self.L](y))
        return y


class OrderPoolingReadout(nn.Module):
    """
    OrderPooling readout
    """

    def __init__(self, input_dim, output_dim, depth, readout_n_layers, readout_pooling, dropout=0.5):  # L=nb_hidden_layers
        super().__init__()
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.mlp_clf = MLPReadout(input_dim * (depth + 1), output_dim, L=int(readout_n_layers), dropout=dropout)
        self.drop = nn.Dropout(dropout)
        if readout_pooling == 'sum':
            self.pool = lambda x: th.sum(x, dim=0)
        elif readout_pooling == 'mean':
            self.pool = lambda x: th.mean(x, dim=0)
        elif readout_pooling == 'max':
            self.pool = lambda x: th.max(x, dim=0)

    def forward(self, h, pos_info):
        emb_list = []
        for i in range(len(pos_info)):
            nodes = pos_info[i]
            if i == 0:  # Center node
                emb_list.append(h[nodes])  # [BSZ, n_hidden]
            else:
                emb_list.append(th.stack([self.pool(h[_.view(-1)]) if _ is not None else
                                          th.zeros_like(h[0]) for _ in nodes]))
        y = self.drop(self.mlp_clf(th.cat(emb_list, dim=1)))
        return y


class OrderPooling(nn.Module):
    """
    OrderPooling for last layer
    """

    def __init__(self, n_class, depth, readout_pooling):  # L=nb_hidden_layers
        super().__init__()
        self.final_clf = nn.Linear(n_class * (depth + 1), n_class, bias=True)
        if readout_pooling == 'sum':
            self.pool = lambda x: th.sum(x, dim=0)
        elif readout_pooling == 'mean':
            self.pool = lambda x: th.mean(x, dim=0)
        elif readout_pooling == 'max':
            self.pool = lambda x: th.max(x, dim=0)

    def forward(self, h, pos_info):
        emb_list = []
        for i in range(len(pos_info)):
            nodes = pos_info[i]
            if i == 0:  # Center node
                emb_list.append(h[nodes])  # [BSZ, n_hidden]
            else:
                emb_list.append(th.stack([self.pool(h[_.view(-1)]) for _ in nodes]))
        y = self.final_clf(th.cat(emb_list, dim=1))
        return y
