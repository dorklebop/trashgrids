from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.nn as tg_nn
from ml_collections import config_dict

from src import nn as src_nn
from src import utils as src_utils
from src.models import modules

import wandb

import gconv.gnn as gnn

class PointCloudResNet(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        in_channels: int,
        out_channels: int,
        domain=[-1.0, 1.0]
    ):
        super().__init__()

        self.grid_representation = Gridifier(
            cfg=cfg, in_channels=in_channels, out_channels=out_channels,
            domain=domain
        )

        self.pool = cfg.net.readout_pool

        if cfg.net.readout_head:
            self.head = nn.Linear(in_features=self.grid_representation.output_channels, out_features=out_channels)
        else:
            self.head = nn.Identity()

  

    def forward(self, data):
        out, out_pos, batch, _ = self.grid_representation(data)

        out = torch.nn.functional.adaptive_avg_pool3d(out, 1)

        batch_size = batch.max() + 1

        out = out.reshape(batch_size, -1)
        
        out = self.head(out)

        return out


class PointCloudSegmentationResNet(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        in_channels: int,
        out_channels: int,
        domain=[-1.0, 1.0]
    ):
        super().__init__()

        self.grid_representation = Gridifier(
            cfg=cfg, in_channels=in_channels, out_channels=out_channels,
            domain=domain
        )

        self.grid_resolution = cfg.gridifier.grid_resolution
        self.num_hidden = cfg.net.num_hidden
        self.num_neighbors = cfg.gridifier.num_neighbors
        self.segmentation_backward_edges = cfg.gridifier.segmentation_backward_edges
        self.reuse_edges = cfg.gridifier.reuse_edges

        position_embed_cfg = cfg.gridifier.position_embed
        update_net_cfg = cfg.gridifier.update_net
        node_embed_cfg = cfg.gridifier.node_embedding
        message_net_cfg = cfg.gridifier.message_net

        self.use_pos_in_output = cfg.gridifier.use_pos_in_output

        self.grid_output_channels = self.grid_representation.output_channels

        self.to_sparse_rep = src_nn.BipartiteConv(
            data_dim=self.grid_representation.dimensions,
            in_channels=self.grid_output_channels,
            out_channels=self.grid_representation.num_hidden,
            num_neighbors=self.grid_representation.num_neighbors,
            conditioning=self.grid_representation.conditioning,
            aggregation=self.grid_representation.aggregation,
            use_target_positions=self.use_pos_in_output,
            use_normals=False,
            position_embed_cfg=position_embed_cfg,
            update_net_cfg=update_net_cfg,
            node_embed_cfg=node_embed_cfg,
            message_net_cfg=message_net_cfg,
        )

        self.grid_out_resolution = self.grid_representation.grid_out_resolution

        self.classifier = nn.Linear(in_features=self.num_hidden, out_features=out_channels)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        out, out_pos, out_batch, edges = self.grid_representation(data)

        # map back to input nodes
        batch_size = batch.max() + 1

        out = (
            out.permute(0, 2, 3, 4, 1)
            .to(memory_format=torch.contiguous_format)
            .reshape(batch_size * (self.grid_out_resolution**3), self.grid_output_channels)
        )

        if self.reuse_edges:
            # transpose adjacency matrix
            neighbor_edges = torch.stack((edges[1], edges[0]))

        else:
            # Build knn connectivity from grid to pointcloud
            forward_edges = tg_nn.knn(
                x=out_pos, y=pos, k=self.num_neighbors, batch_x=out_batch, batch_y=batch
            )
            neighbor_edges = torch.stack((forward_edges[1], forward_edges[0]))

            if self.segmentation_backward_edges:
                # build knn connectivity from pointcloud to grid
                backward_edges = tg_nn.knn(
                    x=pos,
                    y=out_pos,
                    k=self.num_backward_neighbors,
                    batch_x=batch,
                    batch_y=out_batch,
                )
                neighbor_edges = torch.cat((neighbor_edges, backward_edges), dim=-1)

                # remove duplicates
                neighbor_edges = tg.utils.coalesce(neighbor_edges, reduce="mean")

        # Get grid representation
        out, out_pos, out_batch = self.to_sparse_rep(
            edge_index=neighbor_edges,
            x_in=out_pos,
            f_in=out,
            batch_x=out_batch,
            x_out=pos,
            batch_y=batch,
        )

        out = self.classifier(out)

        return out

class EquivariantPointCloudResNet(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        in_channels: int,
        out_channels: int,
        domain=[-1.0, 1.0]
    ):
        super().__init__()

        self.grid_representation = Gridifier(
            cfg=cfg, in_channels=in_channels, out_channels=out_channels,
            domain=domain
        )

        if cfg.net.readout_pool == "mean":
            self.pool = lambda x : torch.nn.functional.adaptive_avg_pool3d(x, 1).squeeze()
        elif cfg.net.readout_pool == "add":
            self.pool = lambda x : torch.sum(x, dim=(2,3,4)).squeeze()
        else:
            self.pool = nn.Identity()

        if cfg.net.readout_head:
#             self.head = nn.Linear(in_features=self.grid_representation.output_channels, out_features=out_channels)
            self.head = nn.Sequential(nn.Linear(in_features=self.grid_representation.output_channels, out_features=cfg.net.num_hidden),
                                      nn.SiLU(),
                                      nn.Linear(in_features=cfg.net.num_hidden, out_features=out_channels))

        else:
            self.head = nn.Identity()

        self.conv_output_size = self.grid_representation.convnet.grid_out_resolution ** 3

        self.flatten_output = cfg.net.flatten_output
        if self.flatten_output:
            conv_output_channels = self.grid_representation.output_channels
#             self.out_shape = lambda x: x.reshape(-1, self.conv_output_size * conv_output_channels)
            self.out_shape = lambda x: x.flatten(1, -1)
            self.head = nn.Sequential(nn.Linear(in_features=self.conv_output_size * conv_output_channels, out_features=cfg.net.num_hidden),
                                    nn.SiLU(),
                                    nn.Linear(in_features=cfg.net.num_hidden, out_features=out_channels))

        else:
            self.out_shape = nn.Identity()



    def forward(self, data, logger=None):
        out, out_pos, batch, _ = self.grid_representation(data, logger=logger)
        out = self.pool(out)
        out = self.out_shape(out)
        out = self.head(out)


        return out



class Gridifier(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict,
        in_channels: int,
        out_channels: int,
        domain=[-1.0, 1.0]
    ):
        super().__init__()


        self.num_hidden = cfg.net.num_hidden

        if cfg.net.norm == "BatchNorm":
            norm_type = f"{cfg.net.norm}3d"
        else:
            norm_type = cfg.net.norm
        norm_class = getattr(nn, norm_type)

        self.use_positions = cfg.dataset.params.use_positions
        self.use_normals = cfg.dataset.params.use_normals

        self.grid_resolution = cfg.gridifier.grid_resolution
        self.num_neighbors = cfg.gridifier.num_neighbors

        if cfg.gridifier.same_k_forward_backward:
            self.num_backward_neighbors = self.num_neighbors
        else:
            self.num_backward_neighbors = cfg.gridifier.num_backward_neighbors
        self.connectivity = cfg.gridifier.connectivity  # TODO(computri): Make this toggleable
        self.conditioning = cfg.gridifier.conditioning
        self.aggregation = cfg.gridifier.aggregation

        position_embed_cfg = cfg.gridifier.position_embed
        update_net_cfg = cfg.gridifier.update_net
        node_embed_cfg = cfg.gridifier.node_embedding
        message_net_cfg = cfg.gridifier.message_net

        embed_nodes = cfg.dataset.md17.embed_features
        self.node_embedding = nn.Identity()

        if embed_nodes:
            self.node_embedding = nn.Embedding(10, self.num_hidden)
            in_channels = self.num_hidden


        # Create grid
        self.domain = domain
        self.dimensions = 3  # TODO(dwromero): cfg.net.data_dim
        grid = src_utils.create_coordinate_grid(
            size=self.grid_resolution,
            domain=self.domain,
            dimensions=self.dimensions,
            as_list=True,
        )
        self.register_buffer("grid", grid)


        self.to_dense_rep = src_nn.BipartiteConv(
            data_dim=self.dimensions,
            in_channels=in_channels,
            out_channels=self.num_hidden,
            num_neighbors=self.num_neighbors,
            conditioning=self.conditioning,
            aggregation=self.aggregation,
            use_target_positions=False,
            use_normals=self.use_normals,
            position_embed_cfg=position_embed_cfg,
            update_net_cfg=update_net_cfg,
            node_embed_cfg=node_embed_cfg,
            message_net_cfg=message_net_cfg,
        )


        if cfg.conv.type == "gconv":
            self.convnet = GConvNet3D(cfg)
        else:
            self.convnet = ConvNet3D(cfg)
        self.output_channels = self.convnet.output_channels


    def forward(self, data, logger=None):

        # Unpack data
        pos, x, batch = data.pos, data.x, data.batch

        batch_size = getattr(data, "num_graphs", 1)

        # target grid and target batch
        batch_y = (
            torch.tensor(list(range(batch_size)), device=pos.device, dtype=torch.long)
            .repeat_interleave(self.grid_resolution**self.dimensions)
            .view(-1)
        )
        target_pos = self.grid.repeat((batch_size, 1)).to(pos.device)

        # Build knn connectivity from grid to pointcloud
        forward_edges = tg_nn.knn(
            x=pos, y=target_pos, k=self.num_neighbors, batch_x=batch, batch_y=batch_y
        )
        neighbor_edges = torch.stack((forward_edges[1], forward_edges[0]))

        if self.num_backward_neighbors:
            # build knn connectivity from pointcloud to grid
            backward_edges = tg_nn.knn(
                x=target_pos, y=pos, k=self.num_backward_neighbors, batch_x=batch_y, batch_y=batch
            )
            neighbor_edges = torch.cat((neighbor_edges, backward_edges), dim=-1)

            # remove duplicates
            neighbor_edges = tg.utils.coalesce(neighbor_edges, reduce="mean")

        x = self.node_embedding(x)

        # Get grid representation
        out, out_pos, batch = self.to_dense_rep(
            edge_index=neighbor_edges,
            x_in=pos,
            f_in=x,
            batch_x=batch,
            x_out=target_pos,
            batch_y=batch_y,
        )

        out = (
            out.reshape(
                batch_size,
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
                self.num_hidden,
            )
            .permute(0, 4, 1, 2, 3)
            .to(memory_format=torch.contiguous_format)
        )

        if logger is not None:
            with torch.no_grad():
                self.plot_grid(out, logger)


        out = self.convnet(out)

        return out, out_pos, batch, neighbor_edges

    def plot_grid(self, grid_rep, logger):
        import torchvision
        import matplotlib.pyplot as plt

        def tile_images(images):
            """
            Tile and display images next to each other using Matplotlib.

            Parameters:
            images (list): A list of image arrays (NumPy arrays).

            Example usage:
            image_array1 = np.random.rand(100, 100, 3)  # Replace with your image arrays
            image_array2 = np.random.rand(100, 100, 3)
            image_array3 = np.random.rand(100, 100, 3)
            images = [image_array1, image_array2, image_array3]
            tile_images(images)
            """
            n = len(images)
            fig, axs = plt.subplots(1, n, figsize=(12, 4))  # You can adjust the figsize as needed

            for i, image_array in enumerate(images):
                minn = image_array.min()
                maxx = image_array.max()
                normed_img = (image_array - minn) / (maxx - minn)
                axs[i].imshow(normed_img)
                axs[i].axis('off')

            plt.tight_layout()

            logger.log({"grid_repr_pre_conv":wandb.Image(fig)})
#             plt.close()

        grid_rep = grid_rep[0, :3, :, :, :].detach().cpu().numpy()

        d_slice = grid_rep.shape[1]
        ims = []
        for d in range(d_slice):
            imslice = grid_rep[:, d, :, :].transpose(1, 2, 0)
            ims.append(imslice)

        tile_images(ims)

class ConvNet3D(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict
    ):
        super().__init__()
        conv_class = getattr(src_nn, cfg.conv.type)
        if conv_class == src_nn.CKConv:
            # Add data_dim and the kernel_config via a partial call
            conv_class = partial(conv_class, data_dim=3, kernel_cfg=cfg.conv.kernel)

        if cfg.net.norm == "BatchNorm":
            norm_type = f"{cfg.net.norm}3d"
        else:
            norm_type = cfg.net.norm
        norm_class = getattr(nn, norm_type)
        nonlinear_class = getattr(nn, cfg.net.nonlinearity)
        dropout_class = getattr(nn, cfg.net.dropout_type)

        self.drop_rate = cfg.net.dropout
        self.num_hidden = cfg.net.num_hidden
        self.num_blocks = cfg.net.num_blocks
        self.grid_resolution = cfg.gridifier.grid_resolution
        self.kernel_size = cfg.conv.kernel.size

        pooling_lyrs = list(map(lambda x: x - 1, cfg.net.pooling_layers))
        block_width_factors = cfg.net.width_factors
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * self.num_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in src_utils.pairwise_iterable(block_width_factors)
            ]
            width_factors = [factor for factor_tuple in width_factors for factor in factor_tuple]
        if len(width_factors) != self.num_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )

        self.grid_out_resolution = self.grid_resolution

        layers = []
        input_channels = output_channels = self.num_hidden
        kernel_size = self.kernel_size
        if self.num_blocks > 0:
            for i in range(self.num_blocks):

                if self.grid_resolution < kernel_size:
                    kernel_size = self.grid_resolution

                if i == 0:
                    input_channels = self.num_hidden
                    output_channels = int(self.num_hidden * width_factors[i])
                else:
                    input_channels = int(self.num_hidden * width_factors[i - 1])
                    output_channels = int(self.num_hidden * width_factors[i])
                layers.append(
                    modules.CKBlock3D(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=self.kernel_size,
                        dropout_rate=self.drop_rate,
                        norm_class=norm_class,
                        conv_class=conv_class,
                        dropout_class=dropout_class,
                        nonlinear_class=nonlinear_class,
                    )
                )
                if i in pooling_lyrs:
                    layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
                    self.grid_out_resolution = self.grid_out_resolution // 2
        else:
            output_channels = self.num_hidden

        self.layers = nn.ModuleList(layers)
        self.output_channels = output_channels

        self.out_norm = norm_class(output_channels)


    def forward(self, x):
        for module in self.layers:
            x = module(x)

        x = self.out_norm(x)
        return x


class GConvNet3D(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict
    ):
        super().__init__()

        if cfg.net.norm == "BatchNorm":
            norm_type = f"{cfg.net.norm}3d"
        else:
            norm_type = cfg.net.norm
        norm_class = getattr(nn, norm_type)
        nonlinear_class = getattr(nn, cfg.net.nonlinearity)
        dropout_class = getattr(nn, cfg.net.dropout_type)

        self.drop_rate = cfg.net.dropout
        self.num_hidden = cfg.net.num_hidden
        self.num_blocks = cfg.net.num_blocks
        self.grid_resolution = cfg.gridifier.grid_resolution
        self.kernel_size = cfg.conv.kernel.size
        self.group_kernel_size = cfg.gconv.group_kernel_size

        self.grid_out_resolution = self.grid_resolution

        layers = []
        input_channels = output_channels = self.num_hidden

        # the lifting layer is required to lift R3 input to the group
        self.lifting_layer = gnn.GLiftingConvSE3(in_channels=self.num_hidden,
                                                 out_channels=self.num_hidden,
                                                 kernel_size=self.kernel_size,
                                                 group_kernel_size=self.group_kernel_size,
                                                 padding="same")
        input_channels = self.num_hidden
        output_channels = self.num_hidden

        if self.num_blocks > 0:
            for i in range(self.num_blocks):

                # HACKS
                if i == (self.num_blocks - 1):
                    output_channels = cfg.conv.out_dim
                # HACKS

                layers.append(
                    modules.GResNetBlock(in_channels=input_channels,
                                         out_channels=output_channels,
                                         padding="same",
                                         kernel_size=self.kernel_size,
                                         group_kernel_size=self.group_kernel_size

                    )
                )

                if i in pooling_lyrs:
                    layers.append(gnn.GMaxSpatialPool3d(kernel_size=2, stride=2))
                    self.grid_out_resolution = self.grid_out_resolution // 2
        else:
            output_channels = self.num_hidden

        self.gpool = gnn.GAvgGroupPool()

        self.layers = nn.ModuleList(layers)
        self.output_channels = output_channels

        self.out_norm = norm_class(output_channels)


    def forward(self, x):
        x, H = self.lifting_layer(x)

        for module in self.layers:
            x, H = module(x, H)

        # pool over group dim
        x = self.gpool(x, H)

        x = self.out_norm(x)

        return x

# class GConvNet3D(nn.Module):
#     def __init__(
#         self,
#         cfg: config_dict.ConfigDict
#     ):
#         super().__init__()
#
#         if cfg.net.norm == "BatchNorm":
#             norm_type = f"{cfg.net.norm}3d"
#         else:
#             norm_type = cfg.net.norm
#         norm_class = getattr(nn, norm_type)
#         nonlinear_class = getattr(nn, cfg.net.nonlinearity)
#         dropout_class = getattr(nn, cfg.net.dropout_type)
#
#         self.drop_rate = cfg.net.dropout
#         self.num_hidden = cfg.net.num_hidden
#         self.num_blocks = cfg.net.num_blocks
#         self.grid_resolution = cfg.gridifier.grid_resolution
#         self.kernel_size = cfg.conv.kernel.size
#         self.group_kernel_size = cfg.gconv.group_kernel_size
#
#         self.grid_out_resolution = self.grid_resolution
#
#         layers = []
#         input_channels = output_channels = self.num_hidden
#
#         # the lifting layer is required to lift R3 input to the group
#         self.lifting_layer = gnn.GLiftingConvSE3(in_channels=self.num_hidden,
#                                                  out_channels=self.num_hidden,
#                                                  kernel_size=self.kernel_size,
#                                                  group_kernel_size=self.group_kernel_size,
#                                                  padding="same")
#         input_channels = self.num_hidden
#         output_channels = self.num_hidden
#
#         if self.num_blocks > 0:
#             for i in range(self.num_blocks):
#
#                 # HACKS
#                 if i == (self.num_blocks - 1):
#                     output_channels = cfg.conv.out_dim
#                 # HACKS
#
#                 layers.append(
#                     gnn.GSeparableConvSE3(in_channels=input_channels,
#                                           out_channels=output_channels,
#                                           padding="same",
#                                           group_kernel_size=self.group_kernel_size,
#                                           kernel_size=self.kernel_size)
#                 )
#
# #                 if i in pooling_lyrs:
# #                     layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
# #                     self.grid_out_resolution = self.grid_out_resolution // 2
#         else:
#             output_channels = self.num_hidden
#
#         self.gpool = gnn.GAvgGroupPool()
#
#         self.layers = nn.ModuleList(layers)
#         self.output_channels = output_channels
#
#         # self.out_norm = norm_class(output_channels)
#
#
#     def forward(self, x):
#         x, H = self.lifting_layer(x)
#
#         for module in self.layers:
#             x, H = module(x, H)
#             x = torch.nn.functional.gelu(x)
#         # pool over group dim
#         x = self.gpool(x, H)
#
#         # x = self.out_norm(x)
#         return x

