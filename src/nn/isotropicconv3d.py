import torch


class IsotropicConv3d(torch.nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(IsotropicConv3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

        self.construct_basis_and_weights()
        self.to(device)

    def construct_basis_and_weights(self):
        # Constructing basis part 1: assign a distance to origin (radius) value to each voxel
        # assuming kernel_size[0]=kernel_size[1]=kernel_size[2]
        grid_1d = torch.arange(-(self.kernel_size[0]//2), self.kernel_size[0]//2 + 1)
        grid_xyz = torch.stack(torch.meshgrid(grid_1d,grid_1d,grid_1d),dim=0)
        radius_grid = grid_xyz.pow(2).sum(0,keepdim=True).sqrt()

        # Construction basis part 2: sample shifted Gaussians as the basis
        num_basis = self.kernel_size[0]//2 + 1  #place one at every integer radius, including 0
        radii = torch.arange(0,num_basis)
        self.register_buffer(name='basis', tensor=(radius_grid - radii[:,None,None,None]).pow(2).neg().exp().div(1.12))

        # Construct the weight parameter: the weights have shape [c_out, c_in/groups, num_basis]
        delattr(self, 'weight')  # remove and redefine the weight parameter
        self.weight = torch.nn.Parameter(torch.zeros(self.out_channels, int(self.in_channels/self.groups), num_basis))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        kernel = torch.tensordot(self.weight , self.basis, dims=([-1],[0]))
        return self._conv_forward(input, kernel, self.bias)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create dirac delta
    B, C, X, Y, Z = 32, 1, 9, 9, 9
    testimage=torch.zeros(B,C,X,Y,Z)
    testimage[:,0,X//2,Y//2,Z//2] = 1.

    # Init the layer
    isconv3Dlayer = IsotropicConv3d(1,1,5, padding='same')

    # Apply the conv and show that the kernel is indeed isotropic
    out = isconv3Dlayer(testimage)
    # The Y-Z plane
    plt.imshow(out[0,0,X//2].detach().numpy())
    plt.show()
    # In the X-Z plane
    plt.imshow(out[0,0,:,Y//2].detach().numpy())
    plt.show()