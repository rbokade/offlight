import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layers.convolution import MatrixEncoder
from modules.layers.graphs import GraphConvolution


def to_tensor(x, device, dtype=th.float32):
    return th.tensor(x, dtype=dtype, device=device)


class ACCommCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCommCritic, self).__init__()
        self.args = args

        self.obs_info = args.obs_info
        input_size = args.obs_info["n_lanes"] + args.obs_info["n_phases"]
        if args.obs_agent_id:
            input_size += args.n_agents
        self.mat_enc = MatrixEncoder(args)

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        # Set up network layers
        device = args.device
        self.adjacency_matrix = to_tensor(args.adj_mat, device)
        self.identity_matrix = th.eye(self.n_agents, device=device)
        self.adj_mat_with_self_conn = th.max(
            self.adjacency_matrix, self.identity_matrix
        )
        self.fc1 = GraphConvolution(input_size, args.hidden_dim)
        self.fc2 = GraphConvolution(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.fc4 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        bs, t, n_agents, _ = inputs.shape
        mat_feats = self.mat_enc(inputs.view(bs * t * n_agents, -1))
        mat_feats = mat_feats.view(bs, t, n_agents, -1)
        res_inputs = inputs[:, :, :, self.obs_info["phase_idxs"][0] :]
        new_inputs = th.cat((mat_feats, res_inputs), dim=-1)
        adj_mat = self.adj_mat_with_self_conn.unsqueeze(0)
        adj_mat = adj_mat.repeat(bs * t, 1, 1)  # (bs, n_agents, n_agents)
        new_inputs = new_inputs.view(bs * t, n_agents, -1)
        x, _ = self.fc1(new_inputs, adj_mat)
        x = F.relu(x)
        x, _ = self.fc2(x, adj_mat)
        x = F.relu(x)
        x = x.view(bs, t, n_agents, -1)
        q_comm = self.fc3(x)
        q_env = self.fc4(x)
        return q_env, q_comm

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, -1, -1)
            )

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            # agent id
            input_shape += self.n_agents
        return input_shape
