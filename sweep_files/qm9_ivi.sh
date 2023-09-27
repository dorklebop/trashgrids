#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00

#SBATCH --output=/home/pvdlind1/gridifier/slurm-%j.out
cd /home/pvdlind1/gridifier/

source activate gridifier

python -m wandb agent ck-experimental/gridifier2/djworgsn
