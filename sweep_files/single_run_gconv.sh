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

# architecture sweep
#python -m wandb agent ck-experimental/DensePointClouds/esm99lov

mkdir -p /scratch-shared/$USER/groups/gridifier
cp -R $HOME/groups/gridifier/* /scratch-shared/$USER/groups/gridifier

cd /scratch-shared/$USER/groups/gridifier

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python main.py --cfg=cfg/qm9_cfg.py --cfg.dataset.augment=True --cfg.gridifier.aggregation=mean --cfg.gridifier.conditioning=distance --cfg.gridifier.grid_resolution=9 --cfg.gridifier.message_net.num_hidden=128 --cfg.gridifier.message_net.num_layers=1 --cfg.gridifier.message_net.type=MLP --cfg.gridifier.node_embedding.num_hidden=128 --cfg.gridifier.node_embedding.num_layers=1 --cfg.gridifier.node_embedding.type= --cfg.gridifier.num_neighbors=7 --cfg.gridifier.position_embed.num_hidden=128 --cfg.gridifier.position_embed.num_layers=1 --cfg.gridifier.position_embed.omega_0=0.1 --cfg.gridifier.update_net.num_hidden=128 --cfg.gridifier.update_net.num_layers=1 --cfg.gridifier.update_net.type=MLP --cfg.net.norm=Identity --cfg.net.num_blocks=3 --cfg.net.num_hidden=32 --cfg.net.pooling_layers="(-1,)" --cfg.optimizer.weight_decay=0 --cfg.train.epochs=150 --cfg.conv.type=gconv --cfg.conv.kernel.size=7 --cfg.gconv.group_kernel_size=12
