from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.nn as tg_nn
from ml_collections import config_dict
import matplotlib.pyplot as plt

from src import nn as src_nn
from src import utils as src_utils
from src.models import modules

import wandb


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

        if cfg.net.readout_pool == "mean":
            self.pool = lambda x : torch.nn.functional.adaptive_avg_pool3d(x, 1).squeeze()
        elif cfg.net.readout_pool == "add":
            self.pool = lambda x : torch.sum(x, dim=(2,3,4)).squeeze()
        else:
            self.pool = nn.Identity()

        if cfg.net.readout_head:
            self.head = nn.Linear(in_features=self.grid_representation.output_channels, out_features=out_channels)
#             self.head = nn.Sequential(nn.Linear(in_features=self.grid_representation.output_channels, out_features=cfg.net.num_hidden),
#                                       nn.SiLU(),
#                                       nn.Linear(in_features=cfg.net.num_hidden, out_features=out_channels))

        else:
            self.head = nn.Identity()



    def forward(self, data, logger=None):
        out, out_pos, batch, _ = self.grid_representation(data, logger=logger)
        out = self.pool(out)
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
            self.node_embedding = nn.Embedding(in_channels, self.num_hidden)
            in_channels = self.num_hidden

        # Create grid
        self.domain = domain
        self.dimensions = 3  # TODO(dwromero): cfg.net.data_dim

        self.circular_grid = cfg.gridifier.circular_grid
        if self.circular_grid:
            grid, circle_idx, _ = src_utils.create_circular_coordinate_grid(
                                                                               size=self.grid_resolution,
                                                                               domain=self.domain,
                                                                               dimensions=self.dimensions,
                                                                               radius=self.domain[1]
                                                                              )


            self.register_buffer("grid", grid)
            self.register_buffer("circle_idx", circle_idx.squeeze())

        else:
            grid = src_utils.create_coordinate_grid(
                size=self.grid_resolution,
                domain=self.domain,
                dimensions=self.dimensions,
                as_list=True,
            )
            self.register_buffer("grid", grid)
        self.grid_size = self.grid_resolution**self.dimensions

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


        self.convnet = ConvNet3D(cfg)
        self.output_channels = self.convnet.output_channels


    def forward(self, data, logger=None):

        # Unpack data
        pos, x, batch = data.pos, data.x, data.batch

        batch_size = getattr(data, "num_graphs", 1)

        if self.circular_grid:
            # target grid and target batch
            batch_y = (
                torch.tensor(list(range(batch_size)), device=pos.device, dtype=torch.long)
                .repeat_interleave(self.circle_idx.shape[0])
                .view(-1)
            )
            circle_indices = (torch.ones(self.circle_idx.shape[0] * batch_size, device=self.grid.device) * self.grid_size) * batch_y + self.circle_idx.tile(batch_size)

            target_pos = self.grid[self.circle_idx].repeat((batch_size, 1)).to(pos.device)
        else:
            # target grid and target batch
            batch_y = (
                torch.tensor(list(range(batch_size)), device=pos.device, dtype=torch.long)
                .repeat_interleave(self.grid_size)
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


        if self.circular_grid:
            rep = torch.zeros(self.grid.shape[0] * batch_size, self.num_hidden, device=self.grid.device)
            rep[circle_indices.long()] = out
            out = rep

        out = (
            out.reshape(
                batch_size,
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
                self.num_hidden,
            )
            .permute(0, 4, 3, 2, 1) #convert from xyz to dhw format
#             .permute(0, 4, 1, 2, 3)
            .to(memory_format=torch.contiguous_format)
        )

        if logger is not None:
            with torch.no_grad():
                self.plot_grid(out, logger)


        out = self.convnet(out)

        return out, out_pos, batch, neighbor_edges

    def plot_grid(self, grid_rep, logger):

        def tile_images(images, minn, maxx):
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

                normed_img = (image_array - minn) / (maxx - minn)
                axs[i].imshow(normed_img)
                axs[i].axis('off')

            plt.tight_layout()

            logger.log({"grid_repr_pre_conv":wandb.Image(fig)})
#             plt.close()

        grid_rep = grid_rep[0, :3, :, :, :].detach().cpu().numpy()

        minn = grid_rep.min()
        maxx = grid_rep.max()

        d_slice = grid_rep.shape[1]
        ims = []
        for d in range(d_slice):
            imslice = grid_rep[:, :, d, :].transpose(1, 2, 0)
            ims.append(imslice)

        tile_images(ims, minn=minn, maxx=maxx)

class ConvNet3D(nn.Module):
    def __init__(
        self,
        cfg: config_dict.ConfigDict
    ):
        super().__init__()


        conv_type = f"{'Isotropic' if cfg.net.kernel.isotropic else ''}Conv3d"
        conv_class = getattr(src_nn, conv_type)
        if conv_class == src_nn.CKConv3d or conv_class == src_nn.IsotropicConv3d:
            # Add data_dim and the kernel_config via a partial call
            conv_class = partial(conv_class, data_dim=3, kernel_cfg=cfg.net.kernel)

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
        self.kernel_size = cfg.net.kernel.size

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

        assert cfg.net.block.type in ["CK", "ConvNeXt"], f"Block type '{cfg.net.block.type}' not recognized. choose ['CK', 'ConvNeXt']"

        block = getattr(modules, f"{cfg.net.block.type}Block3D")

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
                    block(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=self.kernel_size,
                        dropout_rate=self.drop_rate,
                        norm_class=norm_class,
                        conv_class=conv_class,
                        dropout_class=dropout_class,
                        nonlinear_class=nonlinear_class,
                        cfg=cfg
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


