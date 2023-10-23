import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from .transform_utils import get_rotation_matrices



def rotation_invariance_plot(angles, data_0, model):
    rep_0, _, _, _ = model(data_0)
    rep_0 = torch.nn.functional.adaptive_avg_pool3d(rep_0, 1).squeeze()
    transformed_data_list = [rep_0.detach().cpu().numpy()]
    losses = [0.0]


    for angle in angles[1:]:
        R_x, R_y, R_z = get_rotation_matrices(0.0, 0.0, angle, device=data_0.pos.device)
        rot_pos = data_0.pos @ R_x @ R_y @ R_z

        transformed_data = data_0.clone()

        transformed_data.pos = rot_pos

        transformed_rep, _, _, _ = model(transformed_data)

        transformed_rep = torch.nn.functional.adaptive_avg_pool3d(transformed_rep, 1).squeeze()

        transformed_data_list.append(transformed_rep.detach().cpu().numpy())
        losses.append(torch.mean((transformed_rep - rep_0) ** 2).item())
    return losses, angles

def plot_grid(grid_rep, logger, name="post_conv_repr", channels=3):


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

            logger.log({name:wandb.Image(fig)})
#             plt.close()

        grid_rep = grid_rep[0, :channels, :, :, :].detach().cpu().numpy()

        minn = grid_rep.min()
        maxx = grid_rep.max()

        d_slice = grid_rep.shape[1]
        ims = []
        for d in range(d_slice):
            imslice = grid_rep[:, :, d, :].transpose(1, 2, 0)
            ims.append(imslice)

        tile_images(ims, minn, maxx)

#         plt.show()

#     circle = np.linspace(0, 2*np.pi, 100)
#
#
#     fig = plt.figure()
#     plt.plot(np.cos(circle), np.sin(circle), c="red")
#
#     start_x = np.cos(0.0)
#     start_y = np.sin(0.0)
#
#     for loss, angle in zip(losses[1:], angles[1:]):
#
#         x = np.cos(angle) * (1+loss)
#         y = np.sin(angle) * (1+loss)
#
#         plt.plot([start_x, x], [start_y, y], c="blue")
#
#         start_x = x
#         start_y = y

#     plt.show()
#     print(losses)



#
#     plot_post_pool_reps(transformed_data_list)
#     plt.plot(angles, losses)
#     plt.show()



def plot_post_pool_reps(data_list):

            # Create a figure and axis for each subplot
    num_plots = len(data_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))  # 1 row and 'num_plots' columns

    # Plot each 1D array in a separate subplot
    for i, data in enumerate(data_list):

        ax = axes[i]
        ax.imshow(np.reshape(data, (1, -1)))
        ax.set_title(f"Plot {i+1}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

    plt.tight_layout()  # Ensures subplots don't overlap
    plt.show()

def equivariance_loss(graph, model):


        angle = np.random.uniform(0, 2*np.pi, 3)


        angle =  np.pi * 0.5
        R_x, R_y, R_z = get_rotation_matrices(0, -angle, 0, graph.pos.device)

        rot_pos = graph.pos @ R_x @ R_y @ R_z


        graph_rot = graph.clone()


        graph_rot.pos = rot_pos

        grid_res = model.grid_resolution


        out, out_pos, _, _ = model(graph)

#         --------------------------
#         grid_res = self.network.grid_representation.grid_resolution
#         out = torch.zeros(out.shape, device=out.device)
#
#         out[0, :, grid_res//2-1, grid_res//2-2, grid_res//2-1] = 1.0

#         --------------------------
        plot_grid(out, name="f(x)")
#         self.plot_grid(out[:,:1,:,:,:], name="f(x)")

#         print(graph.pos.T)
#         print(graph_rot.pos.T)


        out_rot, out_pos_rot, _, _ = model(graph_rot)

#         self.plot_grid(out_rot, name="f(t(x))")

#         R_x, R_y, R_z = get_rotation_matrices(0, -angle, 0, graph.pos.device)
#         out_pos_rot = out_pos_rot @ R_x @ R_y @ R_z
        out_pos_rot /= 9.9339
        out_pos_rot = out_pos_rot.reshape(1,
                                          grid_res,
                                          grid_res,
                                          grid_res,
                                          -1)


        #         delta
#         grid_res = self.network.grid_representation.grid_resolution
#         out = torch.zeros(out_rot.shape, device=out_rot.device)
#
#         out[0, :, grid_res//2-1, grid_res//2-1, grid_res//2-1] = 1.0

#         print(out.shape)



        out_rot = torch.nn.functional.grid_sample(out_rot, out_pos_rot, mode="nearest", align_corners=True)
#         out_rot = torch.nn.functional.grid_sample(out_rot, out_pos_rot)

        plot_grid(out_rot, self.logger.experiment, name="sampled")
        quit()
#         self.plot_grid_3d(out_pos_rot, out_rot[0, 0, :,:,:].view(-1, 1))

#         self.plot_grid(out_rot[:,:1,:,:,:], name="f(t(x))")

#         quit()

#         R_x, R_y, R_z = get_rotation_matrices(0, 0, -angle, graph.pos.device)
#         inv_rot_pos = out_pos_rot @ R_x @ R_y @ R_z




#         out_pos = out_pos_rot @ R_z.T @ R_y.T @ R_x.T


#         inv_rot_pos /= 9.9339
#
#
#
#         inv_rot_pos = inv_rot_pos.reshape(1,
#                                           grid_res,
#                                           grid_res,
#                                           grid_res,
#                                           -1)

#         out = torch.rot90(out_rot, k=1, dims=[0, 0, 1, 0, 0])
#         out = torch.nn.functional.grid_sample(out[:,:1,:,:,:], inv_rot_pos, mode="nearest", align_corners=True, padding_mode="reflection")
#         out = torch.nn.functional.grid_sample(out_rot, inv_rot_pos, mode="nearest")

#         out_rot = self.rotate_3d(out_rot, -angle * 180 / np.pi)

#         self.plot_grid(out_rot, name="t-1(f(t(x)))")
#         quit()
#
#         eq_error = ((out - out_rot) ** 2).sum() / (out_rot ** 2).sum()
#
#         self.logger.experiment.log(
#                 {
#                     "val/equivariance_loss": eq_error,
#                     "global_step": self.global_step,
#                 }
#             )

def rotate_3d(imgs, angle, axis="z"):
    height = imgs.shape[2]

    img_list = []
    for h in range(height):
        out = torchvision.transforms.functional.rotate(imgs[:,:1, h, :,:], angle=angle)
        img_list.append(out.unsqueeze(2))


    return torch.cat(img_list, dim=2)



def plot_object(f, pos):

    n_nodes = pos.shape[0]
    i = 0
    pos = pos.detach().cpu().numpy()
#         f = f[:, 0].detach().cpu().numpy()
    f = f.detach().cpu().numpy()

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title='',
        scene=dict(
            xaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=True
            ),
            yaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=True
            ),
            zaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=True
            )
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[go.Scatter3d(x=x,
                                       y=y,
                                       z=z,
                                       mode='markers',
                                       marker=dict(size=5,
                                                   color=f,  # set color to an array/list of desired values
                                                   colorscale='rainbow',  # choose a colorscale
                                                   cmax=100,
                                                   cmin=100,
                                                   opacity=0.8)
                                       )],
                    layout=layout)

    fig.show()

def plot_grid_3d(positions, features):
    positions = positions.detach().cpu().numpy()
    features = features.detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata, ydata, zdata = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.scatter3D(xdata, ydata, zdata, c=features, s=20)
    fig.show()



#         return fig

#     def test_step(self, batch, batch_idx):
#         # Perform step
#         predictions, loss = self._step(batch, self.test_mse, time_inference=False)
#         # Log and return loss (Required in training step)
#         self.log(
#             "test/loss",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             sync_dist=self.distributed,
#             batch_size=self.batch_size,
#         )
#         self.log(
#             "test/mse",
#             self.test_mse,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             sync_dist=self.distributed,
#             batch_size=self.batch_size,
#         )
