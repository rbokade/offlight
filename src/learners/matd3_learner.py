import copy
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd


class MATD3Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.actor_freq = args.actor_freq
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic1 = critic_registry[args.critic_type](scheme, args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2 = critic_registry[args.critic_type](scheme, args)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        self.agent_optimiser = Adam(
            params=self.agent_params,
            lr=self.args.lr,
        )
        self.critic_optimiser = Adam(
            params=self.critic_params,
            lr=self.args.lr,
        )

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        # self.last_target_update_step = 0
        self.last_target_update_episode = 0

        self.log_actor = {"actor_loss": [], "actor_grad_norm": []}
        if "bc" in self.args.name:
            self.log_actor["bc_loss"] = []
            self.log_actor["td3_loss"] = []

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            if self.args.use_local_rewards:
                self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        self.importance_sampling = getattr(args, "importance_sampling", False)

    def train(self, batch, t_env: int, episode_num: int):
        critic_log = self.train_critic(batch)

        if (self.training_steps + 1) % self.actor_freq == 0:
            batch_size = batch.batch_size
            critic_inputs = self._build_critic_inputs(batch)
            actions_4bc = batch["actions"][:, :-1]

            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
            mask = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)

            self.mac.init_hidden(batch_size)
            pis = []
            actions = []
            for t in range(batch.max_seq_length - 1):
                pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
                pis.append(pi)
                actions.append(gumbel_softmax(pi, hard=True))

            actions = th.cat(actions, dim=1)
            actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions)
            actions = actions.expand(-1, -1, self.n_agents, -1)

            new_actions = []
            for i in range(self.n_agents):
                temp_action = th.split(actions[:, :, i, :], self.n_actions, dim=2)
                actions_i = []
                for j in range(self.n_agents):
                    if i == j:
                        actions_i.append(temp_action[j])
                    else:
                        actions_i.append(temp_action[j].detach())
                actions_i = th.cat(actions_i, dim=-1)
                new_actions.append(actions_i.unsqueeze(2))
            new_actions = th.cat(new_actions, dim=2)

            q = self.critic1(critic_inputs[:, :-1], new_actions)
            q = q.reshape(-1, 1)

            if self.importance_sampling:
                pis_ = th.cat(pis, dim=1).clone().detach()
                t_probs = th.softmax(pis_, dim=-1)
                t_probs_chosen = th.gather(t_probs, dim=-1, index=actions_4bc).squeeze()
                b_probs = batch["b_probs"][:, :-1]
                b_probs_chosen = th.gather(b_probs, dim=-1, index=actions_4bc).squeeze()
                importance_weights = (t_probs_chosen / (b_probs_chosen + 1e-6)).detach()
                importance_weights = importance_weights / (
                    importance_weights.mean(dim=(1, 2), keepdim=True) + 1e-6
                )
                priorities = batch["priorities"]
                priorities = priorities.unsqueeze(-1)
                priorities = priorities.expand_as(importance_weights)
                beta = getattr(self.args, "priority_beta", 1.0)
                adjusted_weights = importance_weights / (priorities**beta + 1e-6)
                adjusted_weights = adjusted_weights / (adjusted_weights.mean() + 1e-6)
                adjusted_weights = adjusted_weights.reshape(-1, 1)

            if "bc" in self.args.name:
                pis = th.cat(pis, dim=1)
                pis_mask = mask.expand_as(pis)
                pis = pis.reshape(-1, self.n_actions)
                bc_loss = F.cross_entropy(pis, actions_4bc.reshape(-1), reduction="sum")
                # if self.importance_sampling:
                #     priorities = priorities.reshape(-1)
                #     norm_priorities = (priorities - priorities.min()) / (
                #         priorities.max() - priorities.min()
                #     )
                #     bc_loss = (bc_loss * norm_priorities).sum()
                # else:
                #     bc_loss = bc_loss.sum()
                bc_loss = bc_loss / (pis_mask.sum())
                mask = mask.reshape(-1, 1)
                lmbda = self.args.td3_alpha / (
                    (q * mask).abs().sum().detach() / mask.sum()
                )
                if self.importance_sampling:
                    td3_loss = -lmbda * (q * mask * adjusted_weights).mean()
                else:
                    td3_loss = -lmbda * (q * mask).mean()
                actor_loss = td3_loss + self.args.bc_lambda * bc_loss
            else:
                pis = th.cat(pis, dim=1)
                pis[pis == -1e10] = 0
                masked_pis = pis * mask.expand_as(pis)
                masked_pis = masked_pis.reshape(-1, 1)
                mask = mask.reshape(-1, 1)
                if self.importance_sampling:
                    actor_loss = (
                        -(q * mask * adjusted_weights).mean()
                        + self.args.reg * (masked_pis**2).mean()
                    )
                else:
                    actor_loss = (
                        -(q * mask).mean() + self.args.reg * (masked_pis**2).mean()
                    )

            # Optimise agents
            self.agent_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

            if (
                self.args.target_update_interval_or_tau > 1
                and (episode_num - self.last_target_update_episode)
                / self.args.target_update_interval_or_tau
                >= 1.0
            ):
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.log_actor["actor_loss"].append(actor_loss.item())
            self.log_actor["actor_grad_norm"].append(actor_grad_norm.item())
            if "bc" in self.args.name:
                self.log_actor["bc_loss"].append(bc_loss.item())
                self.log_actor["td3_loss"].append(td3_loss.item())
            if self.importance_sampling:
                self.logger.log_stat(
                    "importance_weights_mean", adjusted_weights.mean().item(), t_env
                )
                self.logger.log_stat(
                    "importance_weights_std", adjusted_weights.std().item(), t_env
                )

        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in critic_log.items():
                self.logger.log_stat(k, v, t_env)
            if len(self.log_actor["actor_loss"]) > 0:
                ts = len(self.log_actor["actor_loss"])
                for k, v in self.log_actor.items():
                    self.logger.log_stat(k, sum(v) / ts, t_env)
                    self.log_actor[k].clear()

            self.log_stats_t = t_env

    def train_critic(self, batch):
        critic_log = {}
        batch_size = batch.batch_size

        if self.args.use_local_rewards:
            rewards = batch["local_rewards"][:, :-1]
            bs, t, n, _ = rewards.shape
            rewards = rewards.view(bs, t, n)
        else:
            rewards = batch["reward"][:, :-1]
            rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)

        actions = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.use_local_rewards:
            rewards = rewards.view(bs, t, n, 1)

        critic_inputs = self._build_critic_inputs(batch)
        actions = actions.view(
            batch_size, -1, 1, self.n_agents * self.n_actions
        ).expand(-1, -1, self.n_agents, -1)
        q_taken1 = self.critic1(critic_inputs[:, :-1], actions[:, :-1].detach())
        q_taken2 = self.critic2(critic_inputs[:, :-1], actions[:, :-1].detach())

        q_taken1 = q_taken1.view(batch_size, -1, 1)
        q_taken2 = q_taken2.view(batch_size, -1, 1)

        self.target_mac.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)[:, 1:]

        target_actions = target_actions.view(
            batch_size, -1, 1, self.n_agents * self.n_actions
        ).expand(-1, -1, self.n_agents, -1)
        target_vals1 = self.target_critic1(
            critic_inputs[:, 1:], target_actions.detach()
        )
        target_vals2 = self.target_critic2(
            critic_inputs[:, 1:], target_actions.detach()
        )
        target_vals = th.min(target_vals1, target_vals2)
        target_vals = target_vals.view(batch_size, -1, 1)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = (
            rewards.reshape(-1, 1)
            + self.args.gamma
            * (1 - terminated.reshape(-1, 1))
            * target_vals.reshape(-1, 1).detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error1 = q_taken1.view(-1, 1) - targets.detach()
        td_error2 = q_taken2.view(-1, 1) - targets.detach()
        masked_td_error1 = td_error1 * mask.reshape(-1, 1)
        masked_td_error2 = td_error2 * mask.reshape(-1, 1)
        if self.importance_sampling:
            t_pis = []
            self.mac.init_hidden(batch_size)
            for t in range(batch.max_seq_length):
                pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
                t_pis.append(pi)
            t_pis = th.cat(t_pis, dim=1)[:, :-1]
            t_probs = th.softmax(t_pis.detach(), dim=-1)
            b_probs = batch["b_probs"][:, :-1]
            actions_idxs = batch["actions"][:, :-1]
            t_probs_chosen = th.gather(t_probs, dim=-1, index=actions_idxs).squeeze()
            b_probs_chosen = th.gather(b_probs, dim=-1, index=actions_idxs).squeeze()
            importance_weights = t_probs_chosen / (b_probs_chosen + 1e-6)
            importance_weights = importance_weights / (
                importance_weights.mean(dim=(1, 2), keepdim=True) + 1e-6
            )
            priorities = batch["priorities"]
            priorities = priorities.unsqueeze(-1)
            priorities = priorities.expand_as(importance_weights)
            beta = getattr(self.args, "priority_beta", 1.0)
            adjusted_weights = importance_weights / (priorities**beta + 1e-6)
            adjusted_weights = adjusted_weights / (adjusted_weights.mean() + 1e-6)
            adjusted_weights = th.clamp(adjusted_weights, max=10.0)
            adjusted_weights = adjusted_weights.reshape(-1, 1)
            masked_td_error1 = masked_td_error1 * adjusted_weights
            masked_td_error2 = masked_td_error2 * adjusted_weights

        td_loss = (
            0.5 * (masked_td_error1**2).mean() + 0.5 * (masked_td_error2**2).mean()
        )
        critic_loss = td_loss

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        mask_elems = mask.sum().item() + 1e-6
        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()
        critic_log["td_error1_abs"] = masked_td_error1.abs().sum().item() / mask_elems
        critic_log["td_error2_abs"] = masked_td_error2.abs().sum().item() / mask_elems
        critic_log["q_taken1_mean"] = (q_taken1).sum().item() / mask_elems
        critic_log["q_taken2_mean"] = (q_taken2).sum().item() / mask_elems
        critic_log["target_mean"] = targets.sum().item() / mask_elems
        return critic_log

    def _build_critic_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        inputs.append(
            batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        )

        if self.args.critic_individual_obs:
            inputs.append(batch["obs"][:, ts])

        if self.args.critic_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat(
                    [
                        th.zeros_like(batch["actions_onehot"][:, 0:1]),
                        batch["actions_onehot"][:, :-1],
                    ],
                    dim=1,
                )
            inputs.append(last_actions)

        if self.args.critic_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, self.n_agents, -1)
            )

        inputs = th.cat(inputs, dim=-1)

        return inputs

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic1.cuda()
        self.target_critic1.cuda()
        self.critic2.cuda()
        self.target_critic2.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.critic1.load_state_dict(
            th.load(
                "{}/critic1.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2.load_state_dict(
            th.load(
                "{}/critic2.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
