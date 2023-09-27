#!/bin/bash
#SBATCH --job-name=dense_pc_sweep
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# load modules
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.7.0

source activate gridifier

mkdir -p /scratch-shared/$USER/gridifier2
cp -R $HOME/gridifier2/* /scratch-shared/$USER/gridifier2

cd /scratch-shared/$USER/gridifier2

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python main.py --cfg=cfg/qm9_cfg.py --cfg.gridifier.grid_resolution=9 --cfg.net.num_blocks=6 --cfg.net.num_hidden=512 --cfg.net.pooling_layers="(2,4)"