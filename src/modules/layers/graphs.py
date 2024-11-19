"""
https://github.com/mttga/pymarl_transformers/blob/main/src/modules/layer/transformer.py
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionGRU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionGRU, self).__init__()
        self.rnn = nn.GRUCell(in_features, out_features)

    def forward(self, x, h_in, adj):
        h_in = h_in.view(-1, h_in.size(-1))
        x = th.matmul(adj, x)
        n_neighbors = th.sum(adj, dim=-1, keepdim=True)
        x = x / n_neighbors
        x = x.view(-1, x.size(-1))
        x = self.rnn(x, h_in)
        return x, None


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.bias = bias
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x: (bs, n_nodes, in_features)
        adj: (bs, n_nodes, n_nodes)
        """
        bs, n_n, _ = x.shape
        x = self.w(x)
        x = x.view(bs, n_n, -1)
        x = th.matmul(adj, x)
        return x, adj.unsqueeze(1)


class GIN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GIN, self).__init__()
        self.in_features = in_features
        self.bias = bias
        self.w = nn.Linear(in_features, out_features, bias=bias)
        self.eps = nn.Parameter(th.zeros(1))

    def forward(self, x, adj):
        bs, n_n, _ = x.shape
        x = self.w(x)
        x = x.view(bs, n_n, -1)
        id_mat = th.eye(n_n).unsqueeze(0).expand_as(adj).to(x.device)
        adj_ = adj + id_mat * self.eps
        x = th.matmul(adj_, x)
        return x, adj_.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads, bias=True):
        super(MultiHeadAttention, self).__init__()
        assert out_features % n_heads == 0
        self.n_heads = n_heads
        self.dk = out_features // n_heads
        self.scale = math.sqrt(self.dk)
        self.k = nn.Linear(in_features, out_features, bias=bias)
        self.q = nn.Linear(in_features, out_features, bias=bias)
        self.v = nn.Linear(in_features, out_features, bias=bias)
        # self.combine_heads = nn.Linear(out_features, out_features, bias=bias)
        self._init_weights()

    def forward(self, x, adj, sp_enc=None):
        """
        x: (bs, n_nodes, in_features)
        adj: (bs, n_nodes, n_nodes)
        sp_enc: (n_nodes, n_heads)
        """
        bs, n_n, _ = x.shape
        # Reshape to (bs, n_nodes, n_heads, head_dim)
        key = self.k(x).view(bs, n_n, self.n_heads, self.dk)
        query = self.q(x).view(bs, n_n, self.n_heads, self.dk)
        value = self.v(x).view(bs, n_n, self.n_heads, self.dk)
        # Reshape to (bs, n_heads, n_nodes, head_dim)
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        # Scores shape: (bs, n_heads, n_nodes, n_nodes)
        scores = th.matmul(query, key.transpose(-2, -1)) / self.scale
        if sp_enc is not None:
            sp_enc[range(n_n), range(n_n)] = 0.0
            sp_enc = sp_enc.unsqueeze(0).unsqueeze(0)
            sp_enc = sp_enc.repeat(bs, self.n_heads, 1, 1)
            scores += sp_enc
        adj = adj.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(adj == 0, -1e10)
        attn = F.softmax(scores, dim=-1)  # (bs, n_heads, n_nodes, n_nodes)
        out = th.matmul(attn, value)  # (bs, n_heads, n_nodes, dk)
        out = out.permute(0, 2, 3, 1)  # (bs, n_nodes, dk, n_heads)
        out = out.reshape(bs, n_n, -1)  # (bs, n_nodes, out_features)
        # out = self.combine_heads(out)  # (bs, n_nodes, out_features)
        return out, attn

    def _init_weights(self):
        nn.init.xavier_uniform_(self.k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v.weight, gain=1 / math.sqrt(2))


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1):
        super(GraphMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        # self.norm1 = nn.LayerNorm(in_features)
        self.attention = MultiHeadAttention(in_features, out_features, n_heads)
        # self.norm2 = nn.LayerNorm(in_features)
        # self.ffn = nn.Sequential(
        #     nn.Linear(out_features, out_features // 2),
        #     nn.ReLU(),
        #     nn.Linear(out_features // 2, out_features),
        # )

    def forward(self, h, adj, sp_enc=None):
        """
        h: (bs, n_nodes, in_features)
        adj: (bs, n_nodes, n_nodes)
        """
        h_out, attn_out = self.attention(h, adj, sp_enc)
        # h_out = self.norm1(h + h_out)
        # out = self.ffn(h_out)
        # out = self.norm2(h_out + out)  # residual
        return h_out, attn_out


class Graphormer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        node_features,
        in_degree_matrix,
        out_degree_matrix,
        shortest_path_dist_matrix,
        edge_features,
        n_heads=1,
    ):
        super(Graphormer, self).__init__()
        self.n_heads = n_heads
        self.node_features = node_features
        _, n_nodes, n_edge_features = edge_features.shape
        self.edge_features = edge_features.view(
            n_nodes, n_nodes * n_edge_features
        )

        self.distance_matrix = shortest_path_dist_matrix
        # self.in_degree_matrix = in_degree_matrix
        # self.out_degree_matrix = out_degree_matrix
        self.in_degree_matrix = in_degree_matrix.unsqueeze(1)
        self.out_degree_matrix = out_degree_matrix.unsqueeze(1)
        max_in_degree = in_degree_matrix.max().item()
        max_out_degree = out_degree_matrix.max().item()
        max_hops = self.distance_matrix.max().item()
        # self.positional_encoding = nn.Linear(
        #     self.node_features.shape[-1], in_features, bias=False
        # )
        # self.edge_encoding = nn.Linear(
        #     self.edge_features.shape[-1], in_features, bias=False
        # )
        # self.in_degree_embedding = nn.Embedding(
        #     int(max_in_degree + 1), in_features
        # )
        # self.out_degree_embedding = nn.Embedding(
        #     int(max_out_degree + 1), in_features
        # )
        self.enc_features = th.cat(
            (
                self.node_features,
                self.in_degree_matrix,
                self.out_degree_matrix,
                self.edge_features,
            ),
            dim=1,
        )
        self.node_embedding = nn.Linear(
            self.enc_features.shape[-1], in_features, bias=False
        )
        self.spatial_encoding = nn.Embedding(int(max_hops + 1), 1)
        self.attention = GraphMultiHeadAttention(
            in_features, out_features, n_heads=n_heads
        )

    def forward(self, h, adj):
        """
        h: (bs, n_nodes, in_features)
        adj: (bs, n_nodes, n_nodes)
        """
        # positional_enc = (
        #     self.positional_encoding(self.node_features)
        #     .unsqueeze(0)
        #     .expand_as(h)
        # )
        # in_degree_enc = (
        #     self.in_degree_embedding(self.in_degree_matrix)
        #     .unsqueeze(0)
        #     .expand_as(h)
        # )
        # out_degree_enc = (
        #     self.out_degree_embedding(self.out_degree_matrix)
        #     .unsqueeze(0)
        #     .expand_as(h)
        # )
        # edge_enc = (
        #     self.edge_encoding(self.edge_features).unsqueeze(0).expand_as(h)
        # )
        # h = h + positional_enc + in_degree_enc + out_degree_enc + edge_enc
        node_enc = self.node_embedding(self.enc_features).unsqueeze(0)
        h = h + node_enc
        spatial_enc = self.spatial_encoding(self.distance_matrix).squeeze()
        out, attn_outs = self.attention(h, adj, spatial_enc)
        return out, attn_outs


class AvgGraphPooling:
    def __call__(self, x, keepdim=True):
        """
        x: (bs, n_nodes, in_features)
        """
        return x.mean(dim=1, keepdim=keepdim)


class SumGraphPooling:
    def __call__(self, x, keepdim=True):
        """
        x: (bs, n_nodes, in_features)
        """
        return x.sum(dim=1, keepdim=keepdim)


class DiffPool(nn.Module):
    def __init__(self, in_features, n_clusters, use_gin=True):
        super(DiffPool, self).__init__()
        self.in_features = in_features
        self.n_clusters = n_clusters
        if use_gin:
            self.assignment = GIN(in_features, n_clusters)
            self.transform = GIN(in_features, in_features)
        else:
            self.assignment = GraphConvolution(in_features, n_clusters)
            self.transform = GraphConvolution(in_features, in_features)

    def forward(self, h, adj):
        """
        h: (bs, n_nodes, in_features)
        adj: (bs, n_nodes, n_nodes)
        """
        h_trans, _ = self.transform(h, adj)
        raw_scores, _ = self.assignment(h, adj)
        scores = F.softmax(raw_scores, dim=-1)
        h_pooled = th.matmul(scores.transpose(1, 2), h_trans)
        adj_without_self_loops = adj - th.eye(adj.shape[-1]).to(adj.device)
        adj_pooled = th.matmul(
            th.matmul(scores.transpose(1, 2), adj_without_self_loops), scores
        )
        link_loss = adj_without_self_loops - th.matmul(
            scores, scores.transpose(1, 2)
        )
        link_loss = th.norm(link_loss, p=2)
        ent_loss = (-scores * th.log(scores + 1e-8)).sum(-1).mean(1).sum()
        return h_pooled, adj_pooled, scores, link_loss, ent_loss


def hypernetwork_graph_convolution(
    agent_qs, weights, adj_mat, biases=None, nonlinearity=None, spectral=True
):
    """
    agent_qs: (bs * t, n_agents, 1) or (bs * t , n_agents, embed_dim)
    weights: (bs * t, n_agents, embed_dim)
    adj_mat: (bs * t, n_agents, n_agents)
    biases: (bs * t, 1, embed_dim)
    """
    if spectral:
        laplacian_eig = compute_laplacian_eigenvectors(adj_mat)
        fourier_x = th.matmul(laplacian_eig.transpose(2, 1), agent_qs)
        sup = th.matmul(fourier_x, weights)
        out = th.matmul(laplacian_eig, sup)
    else:
        sup = th.matmul(agent_qs, weights)
        out = th.matmul(adj_mat, sup)
    if biases is not None:
        out = out + biases
    if nonlinearity is not None:
        out = nonlinearity(out)
    return out


def compute_laplacian_eigenvectors(adjacency_matrix):
    # Compute degree matrix
    degree_matrix = th.diag(th.sum(adjacency_matrix, dim=1))
    # Compute Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
    # Compute eigendecomposition of the Laplacian matrix
    eigenvalues, eigenvectors = th.linalg.eigh(laplacian_matrix)
    # Sort eigenvalues and eigenvectors in ascending order
    sorted_indices = th.argsort(eigenvalues, dim=-1)
    # sorted_eigenvalues = th.gather(eigenvalues, -1, sorted_indices)
    sorted_eigenvectors = th.gather(
        eigenvectors,
        -1,
        sorted_indices.unsqueeze(-2).expand(-1, adjacency_matrix.shape[1], -1),
    )
    return sorted_eigenvectors


class GraphConvolutionHypernetwork(nn.Module):
    def __init__(
        self, n_nodes, in_features, embed_dim, n_layers=1, readout_op="avg"
    ):
        super(GraphConvolutionHypernetwork, self).__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.in_features = in_features
        self.proj_w1 = nn.Linear(in_features, embed_dim)
        self.proj_b1 = nn.Linear(in_features, embed_dim)
        self.weights = nn.ModuleList()
        self.biases = nn.ModuleList()
        for _ in range(n_layers):
            w_out = embed_dim**2
            self.weights.append(MLP(in_features, w_out, hidden=embed_dim))
            self.biases.append(MLP(in_features, embed_dim))
        self.proj_w2 = nn.Linear(in_features, n_nodes)
        self.proj_b2 = nn.Linear(in_features, embed_dim)
        self.proj_w3 = nn.Linear(in_features, embed_dim)
        self.proj_b3 = MLP(in_features, 1, hidden=embed_dim)
        if readout_op == "sum":
            self.readout = SumGraphPooling()
        elif readout_op == "avg":
            self.readout = AvgGraphPooling()
        else:
            raise Exception("Invalid graph readout operation")

    def forward(self, x, adj, h_x, clamp=False):
        """
        x: (bs, n_nodes, 1)
        adj: (bs, n_nodes, n_nodes)
        h_x: (bs, 1, hyper_in_features)
        """
        bs = x.size(0)
        w = self.proj_w1(h_x).view(bs, -1, self.embed_dim)
        b = self.proj_b1(h_x).view(bs, -1, self.embed_dim)
        w = F.relu(w) if clamp else th.abs(w)
        x = th.matmul(x, w) + b
        for weights, biases in zip(self.weights, self.biases):
            w = weights(h_x).view(bs, -1, self.embed_dim)
            b = biases(h_x).view(bs, -1, self.embed_dim)
            w = F.relu(w) if clamp else th.abs(w)
            x = hypernetwork_graph_convolution(
                x, w, adj, biases=b, nonlinearity=F.elu
            )
        w = self.proj_w2(h_x).view(bs, self.n_nodes, -1)
        b = self.proj_b2(h_x).view(bs, self.embed_dim, 1)
        w = F.relu(w) if clamp else th.abs(w)
        x = th.matmul(x.transpose(2, 1), w) + b
        w = self.proj_w3(h_x).view(bs, self.embed_dim, 1)
        b = self.proj_b3(h_x).view(bs, -1, 1)
        w = F.relu(w) if clamp else th.abs(w)
        y = th.matmul(x.transpose(2, 1), w) + b
        return y


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden=None):
        super(MLP, self).__init__()
        if hidden is None:
            hidden = out_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
