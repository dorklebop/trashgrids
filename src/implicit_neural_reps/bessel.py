from torch_geometric.nn.models.dimenet import BesselBasisLayer
import torch


class Bessel(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        nonlinear_class: type[torch.nn.Module],
        norm_class: type[torch.nn.Module],
        use_bias: bool,
        num_basis=12,
        cutoff=4.0,
        **kwargs,
    ):
        super().__init__()
        self.layer = BesselBasisLayer(num_radial=num_basis, cutoff=cutoff)

    def forward(self, x):

        return self.layer(x.squeeze())