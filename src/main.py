import numpy as np
import os
import random
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

# import neptune
# from neptune.integrations.sacred import NeptuneObserver

from run import run
from collect_data import run as collect_data
from offline_run import run as offline_run


SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)
logger = get_logger()

# neptune_run = neptune.init_run()
ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    # run the framework
    # run(_run, config, _log)
    if config['run_file'] == 'online':
        run(_run, config, _log)
    elif config['run_file'] == "collect":
        collect_data(_run, config, _log)
    elif config['run_file'] == "offline":
        offline_run(_run, config, _log)
    else: 
        raise NotImplementedError("hhhh")


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def _get_run_file(params):
    # --run/--collect/--
    run_file = ''
    for _i, _v in enumerate(params):
        if _v.startswith('--') and '=' not in _v:
            run_file = _v[2:]
            del params[_i]
            return run_file
    return run_file

if __name__ == "__main__":
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    run_file = _get_run_file(params)
    config_dict['run_file'] = run_file

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    sacred_results_path = f"{map_name}_results"
    results_path = os.path.join(
        dirname(dirname(abspath(__file__))),
    )
    logger.info(f"Saving to FileStorageObserver in {results_path}/sacred.")
    file_obs_path = os.path.join(
        sacred_results_path, f"sacred/{config_dict['name']}"
    )

    # ex.observers.append(
    #     MongoObserver(
    #         db_name="marltscbench",
    #         # url="mongodb://rohitbokade94:uyw-yma9kjd3keu.DNW@Cluster0",
    #     )
    # )  # url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())
    # ex.observers.append(NeptuneObserver(run=neptune_run))

    ex.run_commandline(params)
