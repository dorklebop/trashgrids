#!/bin/bash
#SBATCH --job-name=dense_pc_sweep
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# load modules
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.7.0

source activate gridifier

mkdir -p /scratch-shared/$USER/gridifier
cp -R $HOME/gridifier/* /scratch-shared/$USER/gridifier

cd /scratch-shared/$USER/gridifier

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python -m wandb agent ck-experimental/gridifier2/idwj914w
