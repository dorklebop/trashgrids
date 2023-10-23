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

# architecture sweep
#python -m wandb agent ck-experimental/DensePointClouds/esm99lov

mkdir -p /scratch-shared/$USER/gridifier3
cp -R $HOME/gridifier3* /scratch-shared/$USER/gridifier

cd /scratch-shared/$USER/gridifier3

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python main.py --cfg=cfg/qm9_best.py --cfg.gridifier.circular_grid=True --cfg.net.block.type=CK --cfg.net.kernel.isotropic=True --cfg.net.kernel.type=RBF

