import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ml_collections import config_dict
import math

from src import implicit_neural_reps as inr



class IsotropicConv3d(nn.Module):
    def __init__(self, kernel_cfg, **kwargs):
        super().__init__()
        self.kernel_type = kernel_cfg.type
        if self.kernel_type == "RBF" or self.kernel_type == "":
            self.conv = RBFConv3d(kernel_cfg=kernel_cfg, **kwargs)
        else:
            self.conv = CKDistConv3d(kernel_cfg=kernel_cfg, **kwargs)

    def forward(self, x):
        return self.conv(x)

class RBFConv3d(nn.Conv3d):
    def __init__(self,
                 data_dim: int,
                 kernel_cfg: config_dict.ConfigDict,
                 **kwargs):
        super().__init__(**kwargs)

        self.sigma = 1.12 #TODO: make this a param
        self.construct_basis_and_weights()

    def construct_basis_and_weights(self):
        # Constructing basis part 1: assign a distance to origin (radius) value to each voxel
        # assuming kernel_size[0]=kernel_size[1]=kernel_size[2]
        grid_1d = torch.arange(-(self.kernel_size[0]//2), self.kernel_size[0]//2 + 1)
        grid_xyz = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d),dim=0)
        radius_grid = grid_xyz.pow(2).sum(0,keepdim=True).sqrt()

        # Construction basis part 2: sample shifted Gaussians as the basis
        num_basis = self.kernel_size[0]//2 + 1  #place one at every integer radius, including 0
        radii = torch.arange(0,num_basis)
        self.register_buffer(name='basis', tensor=(radius_grid - radii[:, None, None, None]).pow(2).neg().exp().div(self.sigma))


        # Construct the weight parameter: the weights have shape [c_out, c_in/groups, num_basis]
        delattr(self, 'weight')  # remove and redefine the weight parameter

        self.weight = torch.nn.Parameter(torch.zeros(self.out_channels, int(self.in_channels / self.groups), num_basis))
        torch.nn.init.xavier_uniform_(self.weight)

        # weight = torch.zeros(self.out_channels, int(self.in_channels/self.groups), num_basis)
        # weight = self.init_weights(weight, num_basis, mode="kaiming")
        # self.weight = torch.nn.Parameter(weight)

#         torch.nn.init.xavier_uniform_(weight, gain=0.05)
#         self.weight = self.weight * 0.5

    def init_weights(self, weight, num_basis, gain=1.0, mode="xavier"):
         # attempt to fix the initialisation
        num_input_fmaps = weight.size(1)
        num_output_fmaps = weight.size(0)
        receptive_field_size = self.kernel_size[0] ** 3

        fan_in = num_input_fmaps * receptive_field_size * num_basis
        fan_out = num_output_fmaps * receptive_field_size #* num_basis


        if mode == "xavier":
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                weight = weight.uniform_(-a, a)
            return weight
        else: #assume kaiming
            a = math.sqrt(5.0)
            gain = torch.nn.init.calculate_gain("leaky_relu", a)
            fan = fan_in
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                return weight.uniform_(-bound, bound)







    def forward(self, input):
        kernel = torch.tensordot(self.weight , self.basis, dims=([-1],[0]))

# rbf:
# tensor(57.4505, device='cuda:0')
# tensor(0.1056, device='cuda:0') tensor(-0.1032, device='cuda:0')
# regular conv:
# tensor(-2.2102, device='cuda:0')
# tensor(0.0540, device='cuda:0') tensor(-0.0540, device='cuda:0')

        return self._conv_forward(input, kernel, self.bias)



class CKDistConv3d(nn.Module):
    """Depth separable CKConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: config_dict.ConfigDict,
        kernel_size: int,
        groups: int,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_dim = data_dim
        self.groups = groups

        # Create kernel
        kernel_type = kernel_cfg.type
        kernel_num_hidden = kernel_cfg.num_hidden
        kernel_num_layers = kernel_cfg.num_layers
        kernel_omega_0 = kernel_cfg.omega_0
        kernel_use_bias = kernel_cfg.use_bias


        if kernel_cfg.norm == "BatchNorm":
            norm_type = f"{kernel_cfg.norm}{data_dim}d"
        else:
            norm_type = kernel_cfg.norm
        kernel_norm_class = getattr(nn, norm_type)
        kernel_nonlinear_class = getattr(nn, kernel_cfg.nonlinearity)

        kernel_class = getattr(inr, kernel_type)
        self.kernel_generator = kernel_class(
            data_dim=1,
            out_channels=in_channels,
            num_hidden=kernel_num_hidden,
            num_layers=kernel_num_layers,
            omega_0=kernel_omega_0,
            use_bias=kernel_use_bias,
            norm_class=kernel_norm_class,
            nonlinear_class=kernel_nonlinear_class,
        )
        # Use chang initialize:
        norm_factor = 1.0 / torch.sqrt(torch.tensor(kernel_size**3))
        self.kernel_generator.output_linear.weight.data *= norm_factor

        grid_1d = torch.arange(-(kernel_size//2), kernel_size//2 + 1)
        grid_xyz = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d),dim=0)
        radius_grid = grid_xyz.pow(2).sum(0, keepdim=True).sqrt().permute(1, 2, 3, 0)


        self.register_buffer("grid", radius_grid)

        self.conv_fn = getattr(torch.nn.functional, f"conv{self.data_dim}d")

    def forward(self, x):
        # Compute kernel
        # The kernel must be permuted to be of shape [in_channels, 1, kernel_size, kernel_size, kernel_size] for it to
        # be compatible with a depth-wise convolution implemented with groups=in_channels.
        kernel = self.kernel_generator(self.grid).permute(3, 0, 1, 2).unsqueeze(1)

        # Convolve
        out = self.conv_fn(input=x, weight=kernel, bias=None, padding="same", groups=self.groups)

        return out


if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
    # Create dirac delta
    B, C, X, Y, Z = 32, 1, 9, 9, 9
    testimage=torch.zeros(B,C,X,Y,Z)
    testimage[:,0,X//2,Y//2,Z//2] = 1.

    # Init the layer
    isconv3Dlayer = RBFConv3d(1,1,5, padding='same')

    # Apply the conv and show that the kernel is indeed isotropic
    out = isconv3Dlayer(testimage)
    # The Y-Z plane
    plt.imshow(out[0,0,X//2].detach().numpy())
    plt.show()
    # In the X-Z plane
    plt.imshow(out[0,0,:,Y//2].detach().numpy())
    plt.show()
# Init the layer
#     isconv3Dlayer = IsotropicCKConv3d(1,1,5, padding='same')

#
# if __name__ == "__main__":
#     kernel = CKConv(
#         in_channels=32,
#         out_channels=128,
#         data_dim=3,
#         kernel_cfg=None,
#         kernel_size=7,
#         groups=1)
#
# # Create dirac delta
#     B, C, X, Y, Z = 32, 1, 9, 9, 9
#     testimage=torch.zeros(B,C,X,Y,Z)
#     testimage[:,0,X//2,Y//2,Z//2] = 1.
#
#     out = kernel(testimage)
#
#     plt.imshow(out[0,0,X//2].detach().numpy())
#     plt.show()
#     # In the X-Z plane
#     plt.imshow(out[0,0,:,Y//2].detach().numpy())
#     plt.show()
#     print(kernel)