# Code for OffLight: An Offline Multi-Agent Reinforcement Learning Framework for Traffic Signal Control

## Usage:

### Data Collection:
```
python src/main.py --collect --config=collect_config_file --env-config=env_config_file \
  with offline_data_controller=controller_type \
  model_save_dir=path_to_model_save_dir \
  env_args.map_name=env_name
```

### Running Offline Algorithms:
```
python src/main.py --offline --config=offline_alg_config_file --env-config=env_config_file \
  with env_args.map_name=env_name
```
