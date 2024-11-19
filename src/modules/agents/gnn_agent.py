import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layers.graphs import (
    GraphConvolution,
    GraphConvolutionGRU,
    GraphMultiHeadAttention,
    Graphormer,
)


def to_tensor(x, device, dtype=th.float32):
    return th.tensor(x, dtype=dtype, device=device)


class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(2 * args.hidden_dim, args.n_actions)
        # Select communication graph type
        device = args.device
        self.adjacency_matrix = to_tensor(args.adj_mat, device)
        self.identity_matrix = th.eye(self.n_agents, device=device)
        self.adj_mat_with_self_conn = th.max(
            self.adjacency_matrix, self.identity_matrix
        )
        # self.agent_positions = to_tensor(agent_coords, args.device)
        self.in_degrees = to_tensor(args.in_degrees, device, dtype=th.long)
        self.out_degrees = to_tensor(args.out_degrees, device, dtype=th.long)
        self.edge_features = to_tensor(args.edge_features, device)
        self.distance_matrix = to_tensor(args.distance_matrix, device, dtype=th.long)
        self.gnn_type = args.gnn_type
        assert hasattr(args, "n_passes"), "`n_passes` not found"
        self.n_passes = args.n_passes
        # Define communication graph networks
        if self.gnn_type == "gcn":
            self.comm = GraphConvolution(args.hidden_dim, args.hidden_dim)
        elif self.gnn_type == "gat":
            assert hasattr(args, "n_heads"), "`n_heads` not found"
            self.n_heads = args.n_heads
            self.comm = GraphMultiHeadAttention(
                args.hidden_dim,
                args.hidden_dim,
                n_heads=self.n_heads,
            )
        elif self.gnn_type == "gcn_gru":
            self.comm = GraphConvolutionGRU(args.hidden_dim, args.hidden_dim)
        elif self.gnn_type == "graphormer":
            assert hasattr(args, "n_heads"), "`n_heads` not found"
            self.n_heads = args.n_heads
            self.comm = Graphormer(
                args.hidden_dim,
                args.hidden_dim,
                # self.agent_positions,
                self.in_degrees,
                self.out_degrees,
                self.distance_matrix,
                self.edge_features,
                n_heads=self.n_heads,
            )
        else:
            raise Exception(f"Invalid gnn_type {self.gnn_type}")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, **kwargs):
        """
        inputs_shape: (bs * n_agents, input_dim)
        hidden_state_shape: (bs, n_agents, hidden_dim)
        """
        bs_n, _ = inputs.shape
        bs, n = bs_n // self.n_agents, self.n_agents
        h_in = hidden_state.view(-1, self.args.hidden_dim)
        # Obtain agent encoding
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        # Communication
        adj_mat = self.adj_mat_with_self_conn.unsqueeze(0)
        adj_mat = adj_mat.repeat(bs, 1, 1)  # (bs, n_agents, n_agents)
        h_c = self._communicate(h, adj_mat)
        q = self.fc2(h_c)
        return q, h

    def _communicate(self, x, adj_mat, test=False):
        bs, n = x.shape[0] // self.n_agents, self.n_agents
        if self.gnn_type == "gcn_gru":
            c = th.zeros_like(x)
            x = x.clone().view(bs, n, -1)
        else:
            c = x.clone().view(bs, n, -1)
        for _ in range(self.args.n_passes):
            if self.gnn_type == "gcn_gru":
                c = c.view(-1, self.args.hidden_dim)
                c, _ = self.comm(x, c, adj_mat)
            else:
                c, _ = self.comm(c, adj_mat)
            c = F.relu(c)
        x_c = th.cat((x.view(bs * n, -1), c.view(bs * n, -1)), dim=-1)
        if test:
            return x_c, c
        else:
            return x_c

    def forward_no_message(self, inputs, hidden_state):
        """
        inputs_shape: (bs * n_agents, input_dim)
        hidden_state_shape: (bs, n_agents, hidden_dim)
        """
        bs_n, _ = inputs.shape
        bs, n = bs_n // self.n_agents, self.n_agents
        h_in = hidden_state.view(-1, self.args.hidden_dim)
        # Obtain agent encoding
        x = F.relu(self.fc1(inputs))
        # Communication
        adj_mat = self.adjacency_matrix.unsqueeze(0)
        adj_mat = adj_mat.repeat(bs, 1, 1)  # (bs, n_agents, n_agents)
        q_values_without_senders = []
        inc_msgs = []
        # Disable one sender at a time
        for i in range(n):
            adj_mat_with_disabled_sender = adj_mat.clone()
            adj_mat_with_disabled_sender[:, i, :] = 0
            x_c, c = self._communicate(x, adj_mat_with_disabled_sender, test=True)
            q = self.fc2(x_c)
            q_values_without_senders.append(q)
            inc_msgs.append(c)
        # Disable all senders
        adj_mat_no_senders = th.zeros_like(adj_mat)
        x_c, c = self._communicate(x, adj_mat_no_senders, test=True)
        h = self.rnn(x_c, h_in)
        q = self.fc2(h)
        q_values_without_senders.append(q)
        inc_msgs.append(c)
        # Stack q values
        q_values_without_senders = th.stack(q_values_without_senders, dim=0)
        inc_msgs = th.stack(inc_msgs, dim=0)
        return q_values_without_senders, inc_msgs
