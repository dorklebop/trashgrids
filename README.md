# Learned gridification for efficient point cloud processing

Installation: works for CUDA11.7 + conda and pip

Create env:
```
conda create --name gridifier python=3.10
conda activate gridifier
```

# install torch geometric

for pytorch < 2.0, use:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```
FOR SNELLIUS:
```
python -m pip install torch_scatter==2.1.1+pt20cu117 torch_sparse==0.6.17+pt20cu117 torch_cluster==1.6.1+pt20cu117 torch_spline_conv==1.2.2+pt20cu117 torch_geometric==2.3.1 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

for pytorch 2:
```
python -m pip install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

# Dependencies

For group convolutions, install gconv from https://github.com/ThijsKuipers1995/gconv/


```
python -m pip install pytorch-lightning
python -m pip install matplotlib
python -m pip install wandb
python -m pip install scikit-image
python -m pip install torchmetrics
python -m pip install timm
python -m pip install plotly
python -m pip install natsort
python -m pip install ml-collections
python -m pip install pandas
```

# Training QM9:

Best run so far:

```
python main.py --cfg=cfg/qm9_cfg.py
```
