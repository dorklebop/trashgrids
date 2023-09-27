#!/bin/bash
#SBATCH --job-name=qm9_alpha
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

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python main.py --cfg=cfg/qm9_cfg.py --cfg.conv.type=IsotropicConv3d --cfg.dataset.augment=True --cfg.dataset.qm9.use_cormorant=True --cfg.gridifier.aggregation=mean --cfg.gridifier.conditioning=distance --cfg.gridifier.grid_resolution=9 --cfg.gridifier.message_net.num_hidden=128 --cfg.gridifier.message_net.num_layers=1 --cfg.gridifier.message_net.type=MLP --cfg.gridifier.node_embedding.num_hidden=128 --cfg.gridifier.node_embedding.num_layers=1 --cfg.gridifier.node_embedding.type=MLP --cfg.gridifier.num_neighbors=9 --cfg.gridifier.position_embed.num_hidden=128 --cfg.gridifier.position_embed.num_layers=2 --cfg.gridifier.position_embed.omega_0=1 --cfg.gridifier.update_net.num_hidden=128 --cfg.gridifier.update_net.num_layers=1 --cfg.gridifier.update_net.type=MLP --cfg.net.norm=Identity --cfg.net.num_blocks=10 --cfg.net.num_hidden=256 --cfg.net.pooling_layers="(-1,)" --cfg.optimizer.weight_decay=0 --cfg.train.epochs=500
