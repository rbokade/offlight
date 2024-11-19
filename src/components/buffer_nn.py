import math

import dask.array as da
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset


def normalize(inp):
    return (inp - inp.min(axis=0)) / (inp.max(axis=0) - inp.min(axis=0) + 1e-6)


def gumbel_softmax_sample(logits, temperature):
    noise = th.rand_like(logits)
    gumbel_noise = -th.log(-th.log(noise + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    return y


class DaskDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        if isinstance(self.data_list[0], da.Array):
            return self.data_list[0].shape[0]
        return len(self.data_list[0])

    def __getitem__(self, idx):
        sample = []
        for d in self.data_list:
            if isinstance(d, da.Array):
                d = d[idx].compute()
            sample.append(th.tensor(d, dtype=th.float32))
        return sample


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        device = input.device
        bs, n, _ = input.size()
        identity = th.eye(n, device=device).unsqueeze(0).expand(bs, n, n)
        adj = adj + identity
        degree = th.sum(adj, dim=2)
        degree_inv_sqrt = th.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0.0
        degree_inv_sqrt = degree_inv_sqrt.unsqueeze(2)
        adj_normalized = adj * degree_inv_sqrt * degree_inv_sqrt.transpose(1, 2)
        support = th.matmul(input, self.weight)
        output = th.matmul(adj_normalized, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return "{} (in_features={}, out_features={})".format(
            self.__class__.__name__, self.in_features, self.out_features
        )


class GraphAttentionLayer(nn.Module):
    def __init__(
        self, in_features, out_features, n_heads=1, concat=True, alpha=0.2, bias=True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.alpha = alpha
        self.bias = bias
        self.w = nn.Parameter(th.Tensor(n_heads, in_features, out_features))
        self.a_src = nn.Parameter(th.Tensor(n_heads, out_features, 1))
        self.a_dst = nn.Parameter(th.Tensor(n_heads, out_features, 1))
        if self.bias:
            if self.concat:
                self.b = nn.Parameter(th.Tensor(n_heads * out_features))
            else:
                self.b = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        if self.bias:
            nn.init.zeros_(self.b)

    def forward(self, input, adj):
        bs, n, _ = input.size()
        h = th.matmul(input.unsqueeze(1), self.w)
        e_src = th.matmul(h, self.a_src).squeeze(-1)
        e_dst = th.matmul(h, self.a_dst).squeeze(-1)
        e = e_src.unsqueeze(3) + e_dst.unsqueeze(2)
        e = self.leakyrelu(e)
        zero_vec = -9e15 * th.ones_like(e)
        adj = adj.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attention = th.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        h_prime = th.matmul(attention, h)
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous()
            h_prime = h_prime.view(bs, n, self.n_heads * self.out_features)
            if self.bias:
                h_prime = h_prime + self.b
            return F.elu(h_prime)
        else:
            h_prime = h_prime.mean(dim=1)
            if self.bias:
                h_prime = h_prime + self.b
            return h_prime

    def _prepare_attentional_mechanism_input(self, h):
        bs, n_heads, n, out_features = h.size()
        h_i = h.unsqueeze(3).repeat(1, 1, 1, n, 1)
        h_j = h.unsqueeze(2).repeat(1, 1, n, 1, 1)
        a_input = th.cat([h_i, h_j], dim=-1)
        return a_input

    def __repr__(self):
        return "{}(in_features={}, out_features={}, n_heads={}, concat={}, alpha={})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.n_heads,
            self.concat,
            self.alpha,
        )


class AverageGlobalPool(nn.Module):
    def forward(self, x, adj=None):
        pooled = th.mean(x, dim=-2)
        return pooled


class RepeatGlobalUnpool(nn.Module):
    def __init__(self, num_nodes):
        super(RepeatGlobalUnpool, self).__init__()
        self.num_nodes = num_nodes

    def forward(self, z):
        return z.unsqueeze(-2).repeat(1, 1, self.num_nodes, 1)


class GraphVAE(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agents, latent_dim, adj, device="cpu"):
        super(GraphVAE, self).__init__()
        n_heads = 4
        n_layers = 2
        hidden_dim = 64
        self.validate_after_n_epochs = 10
        self.device = device
        self.adj = adj.to(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.prev_act_embed = nn.Linear(action_dim, hidden_dim)
        self.act_mask_embed = nn.Linear(action_dim, hidden_dim)
        self.agent_id_embed = nn.Embedding(self.n_agents, hidden_dim)
        self.encoder_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.encoder = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.encoder.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim // n_heads,
                    n_heads=n_heads,
                    concat=True,
                    alpha=0.2,
                )
            )
        self.encoder.append(
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=2 * latent_dim,
                n_heads=1,
                concat=False,
                alpha=0.2,
            )
        )
        self.global_pooling = AverageGlobalPool()
        self.global_unpooling = RepeatGlobalUnpool(self.n_agents)
        self.dropout = nn.Dropout(0.2)
        self.decoder = nn.ModuleList()
        # First decoder layer takes concatenated latent and embedded features
        self.decoder.append(
            GraphAttentionLayer(
                in_features=latent_dim + hidden_dim,
                out_features=hidden_dim // n_heads,
                n_heads=n_heads,
                concat=True,
                alpha=0.2,
            )
        )
        for _ in range(n_layers - 1):
            self.decoder.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim // n_heads,
                    n_heads=n_heads,
                    concat=True,
                    alpha=0.2,
                )
            )
        self.decoder.append(
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=action_dim,
                n_heads=1,
                concat=False,
                alpha=0.2,
            )
        )

    def encoder_forward(self, x):
        bs, t, n, _ = x.shape
        x = x.reshape(bs * t, n, _)
        adj = self.adj.unsqueeze(0)
        adj = adj.repeat(bs * t, 1, 1)
        for enc in self.encoder[:-1]:
            x = enc(x, adj)
            # x = F.leaky_relu(x)
        x = self.encoder[-1](x, adj)
        return x.view(bs, t, n, -1)

    def encoder_rnn_forward(self, x):
        bs, t, n, _ = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bs * n, t, -1)
        x, _ = self.encoder_rnn(x)
        x = x.view(bs, n, t, -1)
        return x.permute(0, 2, 1, 3)

    def decoder_forward(self, z, x):
        bs, t, n, _ = z.shape
        x = self.dropout(x)
        x = th.cat((x, z), dim=-1)
        x = x.view(bs * t, n, -1)
        adj = self.adj.unsqueeze(0)
        adj = adj.repeat(bs * t, 1, 1)
        for dec in self.decoder[:-1]:
            x = dec(x, adj)
            # x = F.leaky_relu(x)
        x = self.decoder[-1](x, adj)
        # x = F.log_softmax(x, dim=-1)
        return x.view(bs, t, n, -1)

    def encode(self, o, _a, m, batch):
        bs, t, n, _ = o.shape
        agent_ids = th.arange(self.n_agents).unsqueeze(0).unsqueeze(0)
        agent_ids = agent_ids.expand(bs, t, self.n_agents).to(o.device)
        obs_embed = self.obs_embed(o)
        prev_act_embed = self.prev_act_embed(_a)
        mask_embed = self.act_mask_embed(m)
        id_embed = self.agent_id_embed(agent_ids)
        obs_embed = obs_embed + prev_act_embed + mask_embed + id_embed
        obs_traj = self.encoder_rnn_forward(obs_embed)
        encoded = self.encoder_forward(obs_traj)
        logits = self.global_pooling(encoded)
        mu, logvar = th.split(logits, self.latent_dim, dim=-1)
        return mu, logvar, obs_embed.view(bs, t, n, -1)

    def decode(self, z, obs_embed):
        out = self.decoder_forward(z, obs_embed)
        return out

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, o, _a, m=None, batch=None):
        bs, t, n, _ = o.shape
        mu, logvar, obs_embed = self.encode(o, _a, m, batch)
        z = self.reparameterize(mu, logvar)
        unpooled_z = self.global_unpooling(z)
        a_recon = self.decode(unpooled_z, obs_embed)
        if m is not None:
            a_recon = a_recon * m
        return a_recon, mu, logvar

    def recon_loss(self, a_recon, a):
        a = a.argmax(dim=-1)
        # recon_loss = F.nll_loss(
        #     a_recon.view(-1, a_recon.size(-1)), a.view(-1), reduction="mean"
        # )
        criterion = nn.CrossEntropyLoss(reduction="sum")
        recon_loss = criterion(a_recon.view(-1, a_recon.size(-1)), a.view(-1))
        return recon_loss

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def loss_fn(self, a_recon, a, mu, logvar, kl_weight):
        recon_loss = self.recon_loss(a_recon, a)
        kl_loss = self.kl_loss(mu, logvar)
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_vae(self, data_loader, num_epochs, learning_rate, kl_weight):
        self.to(self.device)
        self.train()
        print(self)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            for (
                batch_o,
                batch__a,
                batch_a,
                batch_o_,
                batch_m,
            ), batch_tensor in data_loader:
                batch_o = batch_o.to(self.device)
                batch__a = batch__a.to(self.device)
                batch_a = batch_a.to(self.device)
                batch_o_ = batch_o_.to(self.device)
                if batch_m is not None:
                    batch_m = batch_m.to(self.device)
                batch_tensor = batch_tensor.to(self.device)
                optimizer.zero_grad()
                a_recon, mu, logvar = self.forward(
                    batch_o, batch__a, batch_m, batch_tensor
                )
                bs, t, n, _ = batch_o.shape
                loss, recon_loss, kl_loss = self.loss_fn(
                    a_recon, batch_a, mu, logvar, kl_weight
                )
                loss = loss / (bs * t * n)
                recon_loss = recon_loss / (bs * t * n)
                kl_loss = kl_loss / (bs * t * n)
                pred_actions = a_recon.argmax(dim=-1)
                true_actions = batch_a.argmax(dim=-1)
                correct = (pred_actions == true_actions).sum().item()
                total_correct += correct
                total_samples += bs * t * n
                loss.backward()
                clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                n_batches += 1
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                + f"Loss: {total_loss / n_batches:.4f} | "
                + f"Recon Loss: {total_recon_loss / n_batches:.4f} | "
                + f"KL Loss: {total_kl_loss / n_batches:.4f} | "
                + f"Train Accuracy: {total_correct / total_samples:.4f}"
            )
            if epoch == 0 or epoch % self.validate_after_n_epochs == 0:
                self.validate(data_loader, kl_weight, epoch, num_epochs)
        print("VAE training complete.")

    def validate(self, dataloader, kl_weight, epoch, num_epochs):
        self.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_kl_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        n_val_batches = 0
        with th.no_grad():
            for (
                batch_o,
                batch__a,
                batch_a,
                batch_o_,
                batch_m,
            ), batch_tensor in dataloader:
                batch_o = batch_o.to(self.device)
                batch__a = batch__a.to(self.device)
                batch_a = batch_a.to(self.device)
                batch_o_ = batch_o_.to(self.device)
                if batch_m is not None:
                    batch_m = batch_m.to(self.device)
                batch_tensor = batch_tensor.to(self.device)
                a_recon, mu, logvar = self.forward(
                    batch_o, batch__a, batch_m, batch_tensor
                )
                bs, t, n, _ = batch_o.shape
                loss, recon_loss, kl_loss = self.loss_fn(
                    a_recon, batch_a, mu, logvar, kl_weight
                )
                loss = loss / (bs * t * n)
                recon_loss = recon_loss / (bs * t * n)
                kl_loss = kl_loss / (bs * t * n)
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_kl_loss += kl_loss.item()
                # Compute validation accuracy
                pred_actions = a_recon.argmax(dim=-1)
                true_actions = batch_a.argmax(dim=-1)
                correct = (pred_actions == true_actions).sum().item()
                total_val_correct += correct
                total_val_samples += bs * t * n
                n_val_batches += 1
        avg_val_loss = total_val_loss / n_val_batches
        avg_val_recon_loss = total_val_recon_loss / n_val_batches
        avg_val_kl_loss = total_val_kl_loss / n_val_batches
        val_accuracy = total_val_correct / total_val_samples
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            + f"Val Loss: {avg_val_loss:.4f} | "
            + f"Val Recon Loss: {avg_val_recon_loss:.4f} | "
            + f"Val KL Loss: {avg_val_kl_loss:.4f} | "
            + f"Val Accuracy: {val_accuracy:.4f}"
        )
        self.train()

    def get_encodings_and_recons(self, data_loader):
        self.eval()
        all_recons = []
        all_encodings = []
        with th.no_grad():
            for (
                batch_o,
                batch__a,
                batch_a,
                batch_o_,
                batch_m,
            ), batch_tensor in data_loader:
                batch_o = batch_o.to(self.device)
                batch_a = batch_a.to(self.device)
                batch__a = batch__a.to(self.device)
                batch_m = batch_m.to(self.device)
                batch_tensor = batch_tensor.to(self.device)
                a_recons, mu, logvar = self.forward(
                    batch_o, batch__a, batch_m, batch_tensor
                )
                z = self.reparameterize(mu, logvar)
                all_recons.append(a_recons.cpu().numpy())
                all_encodings.append(z.cpu().numpy())
        all_recons = np.concatenate(all_recons, axis=0)
        all_encodings = np.concatenate(all_encodings, axis=0)
        all_recons = np.exp(all_recons)
        return all_encodings, all_recons


def custom_collate_fn(batch):
    batch_o, batch__a, batch_a, batch_o_, batch_m = zip(*batch)
    bs = len(batch_o)
    t, n, _ = batch_o[0].shape
    o = th.stack(batch_o, dim=0)  # (batch_size, t, n, obs_dim)
    _a = th.stack(batch__a, dim=0)  # (batch_size, t, n, action_dim)
    a = th.stack(batch_a, dim=0)  # (batch_size, t, n, action_dim)
    o_ = th.stack(batch_o_, dim=0)  # (batch_size, t, n, obs_dim)
    m = th.stack(batch_m, dim=0)  # (batch_size, t, n, action_dim)
    batch_tensor = th.arange(bs).unsqueeze(1).repeat(1, n * t).view(-1).to(o.device)
    return (o, _a, a, o_, m), batch_tensor


def data_to_tensor(data_list):
    tensor_data = []
    for d in data_list:
        tensor_data.append(th.tensor(d, dtype=th.float32))
    return tensor_data
