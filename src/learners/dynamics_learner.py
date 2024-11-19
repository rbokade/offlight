import torch as th
from torch.optim import RMSprop

from components.standarize_stream import RunningMeanStd

from modules.dynamics.mlp import DynamicsModel
from modules.dynamics.rnn import RNNDynamicsModel


class OfflineDynamicsLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0
        if self.args.use_rnn:
            self.dynamics_model = RNNDynamicsModel(scheme, args)
        else:
            self.dynamics_model = DynamicsModel(scheme, args)
        print(self.dynamics_model)
        self.dynamics_model_params = list(self.dynamics_model.parameters())
        self.dynamics_model_optimiser = RMSprop(
            params=self.dynamics_model_params,
            lr=args.lr,
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )

    def train(self, batch, t_env: int, episode_num: int):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        next_obs = batch["obs"][:, 1:]
        next_state = batch["state"][:, 1:]
        next_avail_actions = batch["avail_actions"][:, 1:]
        rewards = -1 * batch["reward"][:, :-1]
        (
            pred_next_obs,
            pred_next_state,
            pred_next_avail_actions,
            pred_reward,
            pred_done,
        ) = self.dynamics_model(batch)
        pred_next_obs = pred_next_obs[:, :-1]
        pred_next_state = pred_next_state[:, :-1]
        pred_next_avail_actions = pred_next_avail_actions[:, :-1]
        pred_reward = pred_reward[:, :-1]
        pred_done = pred_done[:, :-1]
        obs_loss = th.nn.functional.binary_cross_entropy(
            pred_next_obs, next_obs, reduction="none"
        ).sum(-1)
        state_loss = th.nn.functional.binary_cross_entropy(
            pred_next_state,
            next_state.unsqueeze(-2).expand_as(pred_next_state),
            reduction="none",
        ).sum(-1)
        avail_actions_loss = th.nn.functional.binary_cross_entropy(
            pred_next_avail_actions, next_avail_actions.float(), reduction="none"
        ).sum(-1)
        reward_loss = th.nn.functional.binary_cross_entropy(
            pred_reward,
            th.clip(rewards, 0, 1).unsqueeze(-2).expand_as(pred_reward),
            reduction="none",
        ).sum(-1)
        done_loss = th.nn.functional.binary_cross_entropy(
            pred_done,
            batch["terminated"][:, 1:].float().unsqueeze(-2).expand_as(pred_done),
            reduction="none",
        ).sum(-1)
        mask = mask.expand_as(obs_loss)
        obs_loss = (obs_loss * mask).sum() / mask.sum()
        state_loss = (state_loss * mask).sum() / mask.sum()
        avail_actions_loss = (avail_actions_loss * mask).sum() / mask.sum()
        reward_loss = (reward_loss * mask).sum() / mask.sum()
        done_loss = (done_loss * mask).sum() / mask.sum()
        loss = obs_loss + state_loss + avail_actions_loss + reward_loss + done_loss
        self.dynamics_model_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.dynamics_model_params, self.args.grad_norm_clip
        )
        self.dynamics_model_optimiser.step()
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("obs_loss", obs_loss.item(), t_env)
            self.logger.log_stat("state_loss", state_loss.item(), t_env)
            self.logger.log_stat("avail_actions_loss", avail_actions_loss.item(), t_env)
            self.logger.log_stat("reward_loss", reward_loss.item(), t_env)
            self.logger.log_stat("done_loss", done_loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("state_max", next_state.max().item(), t_env)
            self.logger.log_stat("obs_max", next_obs.max().item(), t_env)
            self.logger.log_stat("reward_mean", rewards.mean().item(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.dynamics_model.cuda()

    def save_models(self, path):
        th.save(self.dynamics_model.state_dict(), "{}/dynamics_model.th".format(path))

    def load_models(self, path):
        self.dynamics_model.load_state_dict(
            th.load("{}/dynamics_model.th".format(path))
        )
