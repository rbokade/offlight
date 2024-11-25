import copy
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.nmix import Mixer

import torch as th
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets


class OfflineQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "nmix":
                self.mixer = Mixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(
            params=self.params,
            lr=args.lr,
        )
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0
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
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if self.args.use_local_rewards:
            rewards = batch["local_rewards"][:, :-1]
            bs, t, n, _ = rewards.shape
            rewards = rewards.view(bs, t, n)
        else:
            rewards = batch["reward"][:, :-1]
        if self.importance_sampling:
            b_probs = batch["b_probs"][:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out, dim=1)
        target_mac_out[avail_actions == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            cons_max_q_vals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
            cons_max_q_vals = self.mixer(cons_max_q_vals, batch["state"])

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        if self.args.cal_target == "td_lambda":
            targets = build_td_lambda_targets(
                rewards,
                terminated,
                mask,
                target_max_qvals.detach(),
                self.n_agents,
                self.args.gamma,
                self.args.td_lambda,
            )
        elif self.args.cal_target == "raw":
            targets = (
                rewards
                + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:].detach()
            )
        else:
            raise ValueError("Unknown target calculation type")

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = chosen_action_qvals - targets.detach()
        if self.importance_sampling:
            t_probs = th.softmax(mac_out[:, :-1].clone().detach(), dim=-1)
            t_probs_chosen = th.gather(t_probs, dim=-1, index=actions).squeeze()
            b_probs_chosen = th.gather(b_probs, dim=-1, index=actions).squeeze()
            importance_weights = t_probs_chosen / (b_probs_chosen + 1e-6)
            priorities = batch["priorities"]
            priorities = priorities.unsqueeze(-1)
            priorities = priorities.expand_as(importance_weights)
            beta = getattr(self.args, "priority_beta", 1.0)
            adjusted_weights = importance_weights / (priorities**beta + 1e-6)
            adjusted_weights = adjusted_weights / (adjusted_weights.mean() + 1e-6)
            masked_td_error = (td_error**2) * mask * adjusted_weights
        else:
            masked_td_error = (td_error**2) * mask
        td_loss = masked_td_error.sum() / mask.sum()
        loss = td_loss

        if "cql" in self.args.name:
            if self.args.cql_type == "individual":
                # CQL-error
                logsumexp_q = th.logsumexp(mac_out[:, :-1], dim=3)
                assert logsumexp_q.shape == chosen_action_qvals.shape
                cql_error = logsumexp_q - chosen_action_qvals
                cql_mask = mask.expand_as(cql_error)
                if self.importance_sampling:
                    cql_loss = (
                        cql_error * cql_mask * importance_weights
                    ).sum() / mask.sum()
                else:
                    cql_loss = (cql_error * cql_mask).sum() / mask.sum()
            elif self.args.cql_type == "global_raw":
                sample_actions_num = self.args.raw_sample_actions
                bs = actions.shape[0]
                ts = actions.shape[1]
                n_actions = self.args.n_actions
                n_agents = self.n_agents
                device = "cuda" if self.args.use_cuda else "cpu"
                repeat_avail_actions = th.repeat_interleave(
                    avail_actions[:, :-1].unsqueeze(0),
                    repeats=sample_actions_num,
                    dim=0,
                )  # (san, bs, ts, na, ad)
                total_random_actions = th.randint(
                    low=0,
                    high=n_actions,
                    size=(sample_actions_num, bs, ts, n_agents, 1),
                ).to(
                    device
                )  # (san, bs, ts, na, 1)

                chosen_if_avail = th.gather(
                    repeat_avail_actions, dim=-1, index=total_random_actions
                ).min(-2)[
                    0
                ]  # (san, bs, ts, 1)
                # Repeat mac_out
                repeat_mac_out = th.repeat_interleave(
                    mac_out[:, :-1].unsqueeze(0), repeats=sample_actions_num, dim=0
                )  # (san, bs, ts, na, ad)
                random_chosen_action_qvals = th.gather(
                    repeat_mac_out, dim=-1, index=total_random_actions
                ).squeeze(
                    -1
                )  # (san, bs, ts, na)
                repeat_state = th.repeat_interleave(
                    batch["state"][:, :-1].unsqueeze(0),
                    repeats=sample_actions_num,
                    dim=0,
                )  # (san, bs, ts, sd)
                # Reshape for mixer
                random_chosen_action_qvals = random_chosen_action_qvals.view(
                    bs * sample_actions_num, ts, -1
                )
                if self.mixer is not None:
                    repeat_state = repeat_state.view(bs * sample_actions_num, ts, -1)
                    random_chosen_action_qtotal = self.mixer(
                        random_chosen_action_qvals, repeat_state
                    ).view(
                        sample_actions_num, bs, ts, 1
                    )  # (san, bs, ts, 1)
                    negative_sampling = th.logsumexp(
                        random_chosen_action_qtotal * chosen_if_avail, dim=0
                    )  # (bs, ts, 1)
                else:
                    random_chosen_action_qvals = random_chosen_action_qvals.view(
                        sample_actions_num, bs, ts, n_agents, -1
                    )
                    chosen_if_avail = chosen_if_avail.unsqueeze(-2).expand_as(
                        random_chosen_action_qvals
                    )
                    negative_sampling = th.logsumexp(
                        random_chosen_action_qvals * chosen_if_avail, dim=0
                    )  # (bs, ts, na)
                    mask_ = mask.unsqueeze(-2).expand_as(negative_sampling)
                dataset_expec = chosen_action_qvals  # Already passed through mixer
                if self.importance_sampling:
                    cql_loss = (
                        (negative_sampling - dataset_expec.unsqueeze(-1))
                        * mask_
                        * adjusted_weights.unsqueeze(-1)
                    ).sum() / mask_.sum()
                else:
                    cql_loss = (
                        (negative_sampling - dataset_expec.unsqueeze(-1)) * mask_
                    ).sum() / mask_.sum()
            elif self.args.cql_type == "global_simplified":
                assert cons_max_q_vals[:, :-1].shape == chosen_action_qvals.shape
                cql_error = cons_max_q_vals[:, :-1] - chosen_action_qvals
                if self.importance_sampling:
                    cql_loss = (cql_error * mask * adjusted_weights).sum() / mask.sum()
                else:
                    cql_loss = (cql_error * mask).sum() / mask.sum()
            else:
                raise ValueError("Unknown cql type")
            loss += self.args.cql_alpha * cql_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
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

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            if "cql" in self.args.name:
                self.logger.log_stat("cql_loss", cql_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            if self.importance_sampling:
                self.logger.log_stat(
                    "importance_weights_mean", adjusted_weights.mean().item(), t_env
                )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
