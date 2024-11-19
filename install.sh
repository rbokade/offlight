#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=install
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --output=logs/install.%j.out

module purge
module load anaconda3/2022.05

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate epymarl

cd /home/bokade.r/CityFlow/

pip install .