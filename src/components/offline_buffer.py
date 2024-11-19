"""
https://github.com/zzq-bot/offline-marl-framework-offpymarl/blob/5ffaa7ee23c0de4ea7f82452ae97d92bb5409ae1/src/components/offline_buffer.py
"""

import gc
import os

import dask.array as da
import h5py
import torch as th
import matplotlib.pyplot as plt
import numpy as np

# from numpy.polynomial.polynomial import Polynomial

from torch.utils.data import TensorDataset, DataLoader
from .buffer_nn import GraphVAE, custom_collate_fn, normalize


def data_to_tensor(data_list):
    tensor_data = []
    for d in data_list:
        if isinstance(d, da.Array):
            d = d.compute()
        tensor_data.append(th.tensor(d, dtype=th.float32))
    return tensor_data


############## DataBatch ##############
class OfflineDataBatch:
    def __init__(self, data, batch_size, max_seq_length, device="cpu") -> None:
        self.data = data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length  # None if taken all length
        self.device = device
        for k, v in self.data.items():
            # (batch_size, T, n_agents, *shape)
            # truncate here, interface directly in offlinebuffer
            self.data[k] = v[:, :max_seq_length].to(self.device)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            elif hasattr(self, item):
                return getattr(self, item)
            else:
                raise ValueError(
                    'Cannot index OfflineDataBatch with key "{}"'.format(item)
                )
        else:
            raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))

    def to(self, device=None):
        if device is None:
            device = self.device
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device  # update self.device

    def keys(self):
        return list(self.data.keys())

    def assign(self, key, value):
        if key in self.data:
            assert 0, "Cannot assign to existing key"
        self.data[key] = value


############## OfflineBuffer ##############
class OfflineBufferH5:  # One Task
    def __init__(
        self,
        args,
        map_name,
        qualities,
        data_path="",  # deepest folder
        max_buffer_size=2000,
        device="cpu",
        shuffle=True,
    ) -> None:
        self.args = args
        self.base_data_folder = args.offline_data_folder
        self.map_name = map_name
        self.qualities = qualities
        self.data_path_list = []
        for i, quality_i in enumerate(qualities):  # e.g. "medium_expert"
            data_path_i = data_path[i] if isinstance(data_path, list) else data_path
            self.data_path_list.extend(self.get_data_path(data_path_i, quality_i))

        self.h5_paths_by_quality = {}  # Store .h5 paths grouped by quality
        for i, final_data_path in enumerate(self.data_path_list):
            quality = final_data_path.split("/")[-2]
            if quality not in self.h5_paths_by_quality:
                self.h5_paths_by_quality[quality] = []
            self.h5_paths_by_quality[quality].extend(
                [
                    os.path.join(final_data_path, f)
                    for f in sorted(os.listdir(final_data_path))
                    if f.endswith(".h5")
                ]
            )
        # print(self.h5_paths_by_quality)

        self.max_buffer_size = 100000000 if max_buffer_size <= 0 else max_buffer_size
        self.device = device  # device does not work actually.
        self.shuffle = shuffle
        max_data_size_per_quality = {
            q: self.max_buffer_size // len(self.qualities) for q in self.qualities
        }
        dataset = []
        quality_data_sizes = {quality: 0 for quality in self.qualities}
        for quality, h5_paths in self.h5_paths_by_quality.items():
            for h5_path in h5_paths:
                quality_data = []
                if max_data_size_per_quality[quality] > 0:
                    data = self._read_data(
                        h5_path, max_data_size_per_quality[quality], False
                    )
                    quality_data.append(data)
                    quality_data_size = data[list(data.keys())[0]].shape[0]
                    max_data_size_per_quality[quality] -= quality_data_size
                    quality_data_sizes[quality] += quality_data_size
                    dataset.extend(quality_data)

        for quality, size in quality_data_sizes.items():
            print(f"Loaded {size} samples from quality '{quality}'")

        # self.data = {
        #     k: np.concatenate([v[k] for v in dataset], axis=0)
        #     for k in dataset[0].keys()
        # }
        self.data = {
            k: da.concatenate([da.from_array(v[k]) for v in dataset], axis=0)
            for k in dataset[0].keys()
        }
        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]
        print(f"Buffer size: {self.buffer_size}")
        if shuffle:
            shuffled_idx = np.random.choice(
                self.buffer_size, self.buffer_size, replace=False
            )
            self.data = {k: v[shuffled_idx] for k, v in self.data.items()}

    def get_data_path(self, data_path, quality):
        data_path = os.path.join(data_path, quality)
        if all([".h5" not in f for f in os.listdir(data_path)]):
            # automatically find a folder
            existing_folders = [
                f
                for f in sorted(os.listdir(data_path))
                if os.path.isdir(os.path.join(data_path, f))
            ]
            assert len(existing_folders) > 0
            return [os.path.join(data_path, folder) for folder in existing_folders]
        else:
            return [data_path]

    def _read_data(self, h5_path, max_data_size, shuffle):
        data = {}
        with h5py.File(h5_path, "r") as f:
            for k in f.keys():
                added_data = f[k][:]
                if k not in data:
                    data[k] = added_data
                else:
                    data[k] = np.concatenate((data[k], added_data), axis=0)
        if not shuffle and data[list(data.keys())[0]].shape[0] > max_data_size:
            data = {k: v[-max_data_size:] for k, v in data.items()}

        keys = list(data.keys())
        original_data_size = data[keys[0]].shape[0]
        data_size = min(original_data_size, max_data_size)

        if shuffle:
            shuffled_idx = np.random.choice(
                original_data_size, data_size, replace=False
            )
            data = {k: v[shuffled_idx] for k, v in data.items()}
        return data

    @staticmethod
    def max_t_filled(filled):
        return th.sum(filled, 1).max(0)[0]

    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        # Check if buffer size is smaller than batch_size
        if self.buffer_size < batch_size:
            # Option 2: Adjust batch size to buffer size
            batch_size = self.buffer_size
        sampled_ep_idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        if (
            getattr(self.args, "use_corrected_terminated", False)
            and "corrected_terminated" in sampled_data
        ):
            sampled_data["terminated"] = sampled_data["corrected_terminated"]
        max_ep_t = self.max_t_filled(filled=sampled_data["filled"]).item()
        offline_data_batch = OfflineDataBatch(
            data=sampled_data,
            batch_size=batch_size,
            max_seq_length=max_ep_t,
            device=self.device,
        )
        return offline_data_batch


class OfflineBuffer:
    def __init__(
        self,
        args,
        map_name,
        quality,
        data_path="",  # deepest folder
        max_buffer_size=2000,
        device="cpu",
        shuffle=True,
        importance_sampling=False,
    ) -> None:

        if args.offline_data_type == "h5":
            if importance_sampling:
                self.buffer = ISOfflineBufferH5(
                    args,
                    map_name,
                    quality,
                    data_path,
                    max_buffer_size,
                    device,
                    shuffle,
                )
            else:
                self.buffer = OfflineBufferH5(
                    args,
                    map_name,
                    quality,
                    data_path,
                    max_buffer_size,
                    device,
                    shuffle,
                )
            self.buffer_size = self.buffer.buffer_size
        else:
            raise NotImplementedError(
                "Do not support offline data type: {}".format(args.offline_data_type)
            )

    def can_sample(self, batch_size):
        return self.buffer.can_sample(batch_size)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def sequential_iter(self, batch_size):
        return self.buffer.sequential_iter(batch_size)

    def reset_sequential_iter(self):
        self.buffer.reset_sequential_iter()


class ISOfflineBufferH5(OfflineBufferH5):
    batch_size = 32
    latent_dim = 2
    epochs = 1
    learning_rate = 3e-3
    kl_weight = 1e-4

    def __init__(
        self,
        args,
        map_name,
        qualities,
        data_path="",
        max_buffer_size=2000,
        device="cpu",
        shuffle=False,
        init_priorities=True,
    ) -> None:
        super(ISOfflineBufferH5, self).__init__(
            args,
            map_name,
            qualities,
            data_path,
            max_buffer_size,
            device,
            shuffle=False,
        )
        if init_priorities:
            self.encodings, self.b_probs = self._compute_behavior_probs()
            self.priorities = self._compute_priorities()

    def sample(self, batch_size):
        if self.buffer_size < batch_size:
            batch_size = self.buffer_size
        sampled_ep_idx = np.random.choice(
            self.buffer_size, batch_size, replace=False, p=self.priorities
        )
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        sampled_b_probs = th.tensor(self.b_probs[sampled_ep_idx])
        sampled_priorities = th.tensor(self.priorities[sampled_ep_idx]).unsqueeze(1)
        sampled_data.update({"b_probs": sampled_b_probs})
        sampled_data.update({"priorities": sampled_priorities})
        if (
            getattr(self.args, "use_corrected_terminated", False)
            and "corrected_terminated" in sampled_data
        ):
            sampled_data["terminated"] = sampled_data["corrected_terminated"]
        max_ep_t = self.max_t_filled(filled=sampled_data["filled"]).item()
        offline_data_batch = OfflineDataBatch(
            data=sampled_data,
            batch_size=batch_size,
            max_seq_length=max_ep_t,
            device=self.device,
        )
        return offline_data_batch

    def _compute_discounted_return(self):
        ep, t, _ = self.data["reward"].shape
        reward = self.data["reward"].reshape(ep, t)  # Shape: [ep, t]
        gamma = getattr(self.args, "gamma", 0.99)
        discounted_return = np.zeros_like(reward)
        for i in range(t):
            discounted_return[:, i] = np.sum(
                [gamma**k * reward[:, i + k] for k in range(t - i)], axis=0
            )
        return discounted_return

    def _normalize_returns(self, returns_per_episode, norm_type="linear", alpha=1.0):
        if norm_type == "linear":
            min_return = np.min(returns_per_episode)
            max_return = np.max(returns_per_episode)
            norm_returns = (returns_per_episode - min_return) / (
                max_return - min_return
            ) + 1e-6
        else:
            norm_returns = np.exp(returns_per_episode)
        norm_returns = norm_returns**alpha
        return norm_returns

    def _compute_priorities(self, norm_type="linear"):
        # discounted_return = self._compute_discounted_return()
        # return_per_episode = returns[:, 0]
        return_per_episode = self.data["reward"].sum(axis=1)
        norm_returns = self._normalize_returns(return_per_episode, norm_type="linear")
        # alpha = getattr(self.args, "priority_alpha", 1.0)
        # priorities = (norm_returns + 1e-6) ** alpha
        priorities = norm_returns / np.sum(norm_returns)
        return priorities.squeeze()

    def _compute_behavior_probs(self):
        ep, t, n, _ = self.data["obs"][:, :-1].shape
        obs = normalize(self.data["obs"][:, :-1])
        next_obs = normalize(self.data["obs"][:, 1:])
        actions = self.data["actions_onehot"][:, :-1, :].reshape(ep, t, n, -1)
        mask = self.data["avail_actions"][:, :-1, :].reshape(ep, t, n, -1)
        prev_actions = np.zeros_like(actions)
        prev_actions[:, 1:, :] = actions[:, :-1, :]

        data = [obs, prev_actions, actions, next_obs, mask]
        tensor_data = data_to_tensor(data)
        dataset = TensorDataset(*tensor_data)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

        obs_dim = obs.shape[-1]
        action_dim = actions.shape[-1]
        adj_matrix = th.tensor(self.args.adjacency_matrix, dtype=th.float)

        vae = GraphVAE(
            obs_dim,
            action_dim,
            n,
            self.latent_dim,
            adj_matrix,
            # device=self.device,
            device="cuda",
        )
        vae.train_vae(
            data_loader,
            num_epochs=self.epochs,
            learning_rate=self.learning_rate,
            kl_weight=self.kl_weight,
        )

        test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        encodings, recons = vae.get_encodings_and_recons(test_dataloader)
        encodings = encodings.reshape(-1, encodings.shape[-1])
        recons = np.concatenate(
            (recons, np.expand_dims(recons[:, -1, :, :], 1)), axis=1
        )

        del vae
        del tensor_data
        del dataset
        del data_loader
        del test_dataloader
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

        return encodings, recons


class DataSaver:
    def __init__(self, save_path, logger=None, max_size=2000) -> None:
        self.save_path = save_path
        self.max_size = max_size
        # self.episode_batch = []
        self.data_batch = []
        self.cur_size = 0
        self.part_cnt = 0
        self.logger = logger
        os.makedirs(save_path, exist_ok=True)

    def append(self, data):
        self.data_batch.append(data)  # data \in OfflineDataBatch/EpisodeBatch
        self.cur_size += data[list(data.keys())[0]].shape[0]
        # if len(self.episode_batch) >= self.max_size:
        if self.cur_size >= self.max_size:
            self.save_batch()

    def save_batch(self):
        # if len(self.data_batch) == 0:
        if self.cur_size == 0:
            return
        keys = list(self.data_batch[0].keys())
        data_dict = {k: [] for k in keys}
        for data in self.data_batch:
            for k in keys:
                if isinstance(data[k], th.Tensor):
                    data_dict[k].append(data[k].numpy())
                else:
                    data_dict[k].append(data[k])

        # concatenate e.g. [(x, T, n_agents, *shape), ...] -> [max_size, T, n_agents, *shape]
        data_dict = {k: np.concatenate(v) for k, v in data_dict.items()}
        save_file = os.path.join(self.save_path, "part_{}.h5".format(self.part_cnt))
        with h5py.File(save_file, "w") as file:
            for k, v in data_dict.items():
                file.create_dataset(k, data=v, compression="gzip", compression_opts=9)
        if self.logger is not None:
            self.logger.console_logger.info(
                "Save offline buffer to {} with {} episodes".format(
                    save_file, self.cur_size
                )
            )
        else:
            print(
                "Save offline buffer to {} with {} episodes".format(
                    save_file, self.cur_size
                )
            )
        self.data_batch.clear()
        self.cur_size = 0
        self.part_cnt += 1

    def close(self):
        self.save_batch()
