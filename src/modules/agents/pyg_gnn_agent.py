import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GatedGraphConv
from torch_geometric.utils import add_self_loops

def to_tensor(x, device, dtype=th.float32):
    return th.tensor(x, dtype=dtype, device=device)


class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args
        self.device = args.device

        if args.n_agents == 1:  # centralized
            self.n_agents = args.adjacency_matrix.shape[0]
            self.centralized = True
            self.input_shape = input_shape // self.n_agents
        else:
            self.n_agents = args.n_agents
            self.centralized = False
            self.input_shape = input_shape

        self.adjacency_matrix = to_tensor(args.adj_mat, self.device)
        self.identity_matrix = th.eye(self.n_agents, device=self.device)
        self.adj_mat_with_self_conn = th.max(self.adjacency_matrix, self.identity_matrix)
        self.edge_index = self.adjacency_matrix.nonzero(as_tuple=False).t().to(self.device)
        self.edge_index, _ = add_self_loops(self.edge_index, num_nodes=self.n_agents)
        self.gnn_type = args.gnn_type
        self.n_passes = args.n_passes

        # Define communication graph networks
        if self.gnn_type == "gcn":
            self.comm = GCNConv(self.input_shape, args.hidden_dim, add_self_loops=True)
        elif self.gnn_type == "gat":
            assert hasattr(args, "n_heads"), "`n_heads` not found"
            self.comm = GATv2Conv(self.input_shape, args.hidden_dim, heads=args.n_heads, add_self_loops=True)
            self.fc_attn_agg = nn.Linear(args.hidden_dim * args.n_heads, args.hidden_dim)
        # elif self.gnn_type == "gcn_gru":
        #     self.comm = GatedGraphConv(args.hidden_dim, num_layers=self.n_passes)
        #     self.fc_attn_agg = nn.Linear(args.hidden_dim * args.n_heads, args.hidden_dim)
        else:
            raise Exception(f"Invalid gnn_type {self.gnn_type}")

        if self.centralized:
            self.rnn = nn.GRUCell(self.n_agents * args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self, batch_size=1):
        return th.zeros(batch_size, self.args.hidden_dim, device=self.device)

    def forward(self, inputs, hidden_state, test=False):
        """
        inputs_shape: (bs * n_agents, input_dim)
        hidden_state_shape: (bs, n_agents, hidden_dim)
        """
        if self.centralized:
            bs = inputs.shape[0]
            # n = self.n_agents
            # inputs = inputs.view(bs * n, self.input_shape)
        else:
            bs_n, _ = inputs.shape
            bs, n = bs_n // self.n_agents, self.n_agents
        h_in = hidden_state.view(-1, self.args.hidden_dim)
        x_c, attn = self._communicate(inputs, self.edge_index, test=test)
        if self.centralized:
            x_c = x_c.view(bs, -1)
        h = self.rnn(x_c, h_in)
        q = self.fc2(h)
        if not test:
            return q, h
        else:
            return q, h, attn

    def _communicate(self, x, edge_index, test=False):
        if not self.centralized:
            bs, n = x.shape[0] // self.n_agents, self.n_agents
        else:
            bs, n = x.shape[0], self.n_agents
        if self.gnn_type == "gcn_gru":
            x = x.view(bs, n, -1)
        else:
            if self.centralized:
                x = x.view(bs, n, -1)
            x = x.view(bs * n, -1)
        attentions = None
        for _ in range(self.n_passes):
            if self.gnn_type == "gat":
                x, attentions = self.comm(x, edge_index, return_attention_weights=True)
            else:
                x = self.comm(x, edge_index)
            x = F.relu(x)
            if self.gnn_type == "gat":
                x = self.fc_attn_agg(x)
        if test:
            return x, attentions 
        else:
            return x, None

    def forward_no_message(self, inputs, hidden_state, edge_index):
        """
        inputs_shape: (bs * n_agents, input_dim)
        hidden_state_shape: (bs, n_agents, hidden_dim)
        """
        bs_n, _ = inputs.shape
        bs, n = bs_n // self.n_agents, self.n_agents
        h_in = hidden_state.view(-1, self.args.hidden_dim)
        x = F.relu(self.fc1(inputs))
        q_values_without_senders = []
        inc_msgs = []
        # Disable one sender at a time
        for i in range(n):
            edge_index_disabled = edge_index[:, edge_index[0] != i]
            x_c, _ = self._communicate(x, edge_index_disabled, test=True)            
            q = self.fc2(x_c)
            q_values_without_senders.append(q)
            inc_msgs.append(x_c)
        # Disable all senders
        edge_index_no_senders = th.empty((2, 0), dtype=th.long, device=self.device)
        x_c, _ = self._communicate(x, edge_index_no_senders, test=True)
        h = self.rnn(x_c, h_in)
        q = self.fc2(h)
        q_values_without_senders.append(q)
        inc_msgs.append(x_c)
        # Stack q values
        q_values_without_senders = th.stack(q_values_without_senders, dim=0)
        inc_msgs = th.stack(inc_msgs, dim=0)
        return q_values_without_senders, inc_msgs
