"""
https://github.com/zzq-bot/offline-marl-framework-offpymarl/blob/5ffaa7ee23c0de4ea7f82452ae97d92bb5409ae1/src/collect_data.py
"""

import datetime
import os
import json
import pprint
import time
import threading
import torch as th
from tqdm import tqdm

from os.path import dirname, abspath
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str

from runners import REGISTRY as r_REGISTRY

from controllers import REGISTRY as mac_REGISTRY

from components.episode_buffer import ReplayBuffer
from components.offline_buffer import DataSaver
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # check sanity w.r.t data quality related params
    # only for sc2 so far
    assert args.evaluate
    if args.offline_data_controller.lower():
        args.t_max = -1  # do not do while loop training
        # args.mac = TRADITIONAL_CONTROLLERS[args.offline_data_controller.lower()]
        args.learner = "null_learner"
    else:
        raise ValueError(
            "Dataset quality {} not supported!".format(args.offline_data_controller)
        )

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = os.path.join(
        dirname(dirname(abspath(__file__))),
        f"{args.env_args['map_name']}_results",
        "collected_data",
    )
    os.makedirs(results_save_dir, exist_ok=True)
    args.results_save_dir = results_save_dir

    # if args.use_wandb:
    #     args.use_tensorboard = False
    # assert args.use_tensorboard and args.use_wandb

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, "logs")
        logger.setup_tb(tb_exp_direc)

    # if args.use_wandb and not args.evaluate:
    #     wandb_run_name = args.results_save_dir.split('/')
    #     wandb_run_name = "/".join(wandb_run_name[wandb_run_name.index("results")+1:])
    #     wandb_exp_direc = os.path.join(results_save_dir, 'logs')
    #     logger.setup_wandb(wandb_exp_direc, project=args.wandb_project_name, name=wandb_run_name,
    #                        run_id=args.resume_id, config=args)

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)
    # set model save dir
    if getattr(args, "model_save_dir", None) is None:
        args.model_save_dir = os.path.join(results_save_dir, "models")

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)
    # runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # evaluate_sequential(args, runner, logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner, logger):
    map_name = args.env_args["map_name"]
    unique_token = f"{args.name}_seed{args.seed}_{map_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    args.unique_token = unique_token
    # save num_episode_collected episodes using loaded agent model
    logger.console_logger.info(
        f"Do offline data collection with collecting {args.num_episodes_collected} trajectories with {args.offline_data_controller}"
    )
    save_path = os.path.join(
        args.results_save_dir, args.offline_data_controller, args.unique_token
    )
    offline_saver = DataSaver(save_path, logger, args.num_episodes_collected)

    logger.log_stat("episode", 0, runner.t_env)
    with th.no_grad():
        for _ in tqdm(range(args.num_episodes_collected)):
            episode_batch = runner.run(
                test_mode=True, controller=args.offline_data_controller
            )
            offline_saver.append(
                data={
                    k: episode_batch[k].clone().cpu()
                    for k in episode_batch.data.transition_data.keys()
                }
            )
    # print recent status
    runner._log(runner.test_returns, runner.test_stats, prefix="collect_data_")
    logger.print_recent_stats()

    # save left samples
    offline_saver.close()

    # close
    # logger.console_logger.info("Save offline buffer to {}".format(offline_saver.save_path))
    runner.close_env()
    logger.console_logger.info("Finished Evaluation")


def run_sequential(args, logger):
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    # Set up schemes and groups here
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "network_flow": {"vshape": (1,)},
        "network_density": {"vshape": (1,)},
    }
    scheme.update(
        {
            "local_rewards": {
                "vshape": (1,),
                "group": "agents",
                "dtype": th.float32,
            }
        }
    )
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    mac = DummyMAC()
    if args.offline_data_controller == "expert":
        buffer = ReplayBuffer(
            scheme,
            groups,
            20,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )
        mac = mac_REGISTRY[args.controller_mac](buffer.scheme, groups, args)
        mac.load_models(args.model_save_dir)

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    evaluate_sequential(args, runner, logger)


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


class DummyMAC:
    def __init__(self):
        self.action_selector = None

    def init_hidden(self, batch_size):
        pass
