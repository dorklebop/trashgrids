import numpy as np
import torch

def get_rotation_matrices(angle_x, angle_y, angle_z, device):
            R_x = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, np.cos(angle_z), -np.sin(angle_z)],
                                [0.0, np.sin(angle_z),  np.cos(angle_z)]], dtype=torch.float32, device=device)

            R_y = torch.tensor([[np.cos(angle_y), 0.0, np.sin(angle_y)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angle_y), 0.0, np.cos(angle_y)]], dtype=torch.float32, device=device)

            R_z = torch.tensor([[np.cos(angle_x), -np.sin(angle_x), 0.0],
                                [np.sin(angle_x),  np.cos(angle_x), 0.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

            return R_x, R_y, R_z