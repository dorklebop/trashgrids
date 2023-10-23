import torch


def create_coordinate_grid(size: int, dimensions: int, domain: tuple[float, float], as_list: bool):
    """Constructs a coordinate grid of size :size: over the domain (x1, x2) with the input
    dimensionality.

    If as_list is True, the grid is returned as a list of coordinates, otherwise as a tensor of
    shape (size, size, ..., size, dimensions).
    """

    if len(domain) != 2:
        raise ValueError(f"The domain must be a tuple of two numbers. Current: {domain}")
    if domain[0] >= domain[1]:
        raise ValueError(
            f"The first value in the domain must be smaller than the next. Current: {domain}"
        )

    linspace = torch.linspace(domain[0], domain[1], size)
    mesh = torch.meshgrid(*(linspace,) * dimensions, indexing="ij")
    if as_list:
        mesh = torch.stack([mesh_i.reshape(-1) for mesh_i in mesh], dim=-1)
    else:
        mesh = torch.stack(mesh, dim=-1)
    return mesh


def create_circular_coordinate_grid(size: int, dimensions: int, domain: tuple[float, float], radius=float):
    """Constructs a coordinate grid of size :size: over the domain (x1, x2) with the input
    dimensionality. Additionally returns the indices of the grid of all the points within a sphere of size radius,
    and the indices outside of it.

    """

    if len(domain) != 2:
        raise ValueError(f"The domain must be a tuple of two numbers. Current: {domain}")
    if domain[0] >= domain[1]:
        raise ValueError(
            f"The first value in the domain must be smaller than the next. Current: {domain}"
        )

    linspace = torch.linspace(domain[0], domain[1], size)
    mesh = torch.meshgrid(*(linspace,) * dimensions, indexing="ij")
    mesh = torch.stack([mesh_i.reshape(-1) for mesh_i in mesh], dim=-1)


    norms = torch.linalg.norm(mesh, dim=-1)
    indices_in = (norms <= radius).nonzero(as_tuple=False)
    indices_out = (norms > radius).nonzero(as_tuple=False)

    return mesh, indices_in, indices_out



if __name__ == "__main__":
   mesh, indices_in, indices_out = create_circular_coordinate_grid(size=100, dimensions=2, domain=(-1.0, 1.0), radius=1.0)

