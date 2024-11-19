#!/bin/bash

envs=("jinan_3_4")
algs=("iql", "vdn", "qmix", "ia2c", "maa2c", "maddpg")

for env in "${envs[@]}"; do
  for alg in "${algs[@]}"; do
    echo "Running with alg=$alg, env=$env"
    python src/main.py --config=$alg --env-config=$env \
      env_args.map_name=$env \
  done
done