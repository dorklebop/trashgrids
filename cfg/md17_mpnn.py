from cfg import default_config


def get_config():
    # Load the default config & then change the variables
    cfg = default_config.get_config()

    cfg.debug = False
    cfg.task = "regression"
    cfg.net.type = "MPNN"
    # dataset
    cfg.dataset.augment = False
    cfg.dataset.name = "MD17"
    cfg.dataset.params.num_classes = 1
    cfg.dataset.params.use_positions = False
    cfg.dataset.md17.embed_features = True
    cfg.dataset.md17.add_edges = True

    cfg.wandb.entity = "ck-experimental"
    cfg.wandb.project = "dense_point_clouds"


    # gridifier
    cfg.gridifier.aggregation = "mean"
    cfg.gridifier.conditioning = "distance"
    cfg.gridifier.connectivity = "knn"
    cfg.gridifier.grid_resolution = 9
    cfg.gridifier.message_net.nonlinearity = "GELU"

    cfg.gridifier.num_neighbors = 9
    # no message net
    cfg.gridifier.message_net.type = "MLP"
    cfg.gridifier.message_net.num_hidden = 128
    cfg.gridifier.message_net.num_layers = 1


    cfg.gridifier.node_embedding.nonlinearity = "GELU"
    cfg.gridifier.node_embedding.num_hidden = 128
    cfg.gridifier.node_embedding.num_layers = 1
    cfg.gridifier.node_embedding.type = "MLP"

    cfg.gridifier.position_embed.nonlinearity = "GELU"
    cfg.gridifier.position_embed.num_hidden = 128
    cfg.gridifier.position_embed.num_layers = 2
    cfg.gridifier.position_embed.omega_0 = 1.0
    cfg.gridifier.position_embed.type = "RFNet"

    cfg.gridifier.update_net.nonlinearity = "GELU"
    cfg.gridifier.update_net.num_hidden = 64
    cfg.gridifier.update_net.num_layers = 1
    cfg.gridifier.update_net.type = "MLP"
    cfg.gridifier.reuse_edges = True
    cfg.gridifier.same_k_forward_backward = True


    # net
    cfg.net.dropout = 0
    cfg.net.nonlinearity = "GELU"
    cfg.net.readout_pool = "mean"
    cfg.net.readout_head = True
    cfg.net.norm = "Identity"
    cfg.net.num_blocks = 6
    cfg.net.num_hidden = 256
    cfg.num_workers = 1

    cfg.conv.type = "IsotropicConv3d"
    cfg.conv.out_dim = 64
    cfg.conv.kernel.size = 5

    cfg.optimizer.lr = 0.0003
    cfg.optimizer.type = "Adam"
    cfg.optimizer.weight_decay = 0.0
    cfg.scheduler.type = "cosine"
    cfg.scheduler.warmup_epochs = 0
    cfg.scheduler.mode = "min"
    cfg.seed = 42
    cfg.train.batch_size = 5
    cfg.train.epochs = 200

    return cfg