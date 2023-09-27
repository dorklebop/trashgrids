# torch
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
# from torch_geometric.loader import DataLoader
import torch_geometric.transforms as tg_transforms
from tqdm import tqdm

from torch_geometric import loader as tg_loader

from typing import Tuple, Dict
from argparse import Namespace


class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        target="alpha",
        **kwargs,
    ):
        super().__init__()

        num_classes = cfg.params.num_classes
        use_positions = cfg.params.use_positions
        use_normals = cfg.params.use_normals
        data_dir = cfg.data_dir
        num_samples = cfg.params.num_samples
        val_split_as_test_split = cfg.params.val_split_as_test_split


        # Save parameters to self
        self.data_dir = data_dir + f"/QM9_alpha"
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.use_positions = use_positions
        self.use_normals = use_normals
        self.val_split_as_test_split = val_split_as_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.min_pos = -9.9339
        self.max_pos = 9.9339

        self.data_dim = 3

        self.cormorant = cfg.qm9.use_cormorant
        if self.cormorant:
            self.input_channels = 15
        else:
            self.input_channels = 11
        self.output_channels = self.num_classes

        transform = [tg_transforms.Center()]

        self.augment = cfg.augment
        if self.augment:
            self.train_transform = tg_transforms.Compose(transform +
                                                         [tg_transforms.RandomFlip(axis=0),
                                                          tg_transforms.RandomFlip(axis=1),
                                                          tg_transforms.RandomFlip(axis=2),
                                                          tg_transforms.RandomRotate(180, axis=0),
                                                          tg_transforms.RandomRotate(180, axis=1),
                                                          tg_transforms.RandomRotate(180, axis=2)])
            self.test_transform = tg_transforms.Compose(transform)
        else:
            self.train_transform = self.test_transform = tg_transforms.Compose(transform)

        self.pre_transform = None

        self.target = target

        self.qm9_to_ev = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        self.targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']

    def prepare_data(self):
        if not self.augment:
            dataset = QM9(root=self.data_dir, transform=self.train_transform, pre_transform=None)
            index = self.targets.index(self.target)
            self.dataset = [prep_data(graph, index, self.target, self.qm9_to_ev, self.cormorant) for graph in tqdm(dataset, desc='Preparing data')]


            # train/val/test split
            n_train, n_test = 100000, 110000


            mean, mad = calc_mean_mad(self.dataset[:n_train])
            self.mean, self.mad = mean.item(), mad.item()

            self.train_dataset = self.dataset[:n_train]

            self.val_dataset = self.dataset[n_test:]

            self.test_dataset = self.dataset[n_train:n_test]


        else:
            dataset_train = QM9(root=self.data_dir, transform=self.train_transform, pre_transform=None)
            dataset_test = QM9(root=self.data_dir, transform=self.test_transform, pre_transform=None)
            index = self.targets.index(self.target)
            self.dataset_train = [prep_data(graph, index, self.target, self.qm9_to_ev, self.cormorant) for graph in tqdm(dataset_train, desc='Preparing train data')]
            self.dataset_test = [prep_data(graph, index, self.target, self.qm9_to_ev, self.cormorant) for graph in tqdm(dataset_test, desc='Preparing test data')]


            # train/val/test split
            n_train, n_test = 100000, 110000


            mean, mad = calc_mean_mad(self.dataset_train[:n_train])
            self.mean, self.mad = mean.item(), mad.item()


            self.train_dataset = self.dataset_train[:n_train]

            self.val_dataset = self.dataset_test[n_test:]

            self.test_dataset = self.dataset_test[n_train:n_test]


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        pass



    def train_dataloader(self):
        return tg_loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return tg_loader.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return tg_loader.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

def prep_data(graph: Data, index: int, target_name: str, qm9_to_ev: Dict[str, float], cormorant=False) -> Data:

    graph.y = graph.y[0, index].unsqueeze(-1).unsqueeze(-1)

    # change unit of targets
    if target_name in qm9_to_ev:
        graph.y *= qm9_to_ev[target_name]

    if cormorant:
        one_hot = graph.x[:, :5]
        z = graph.z
        max_scale = z.max()
        graph.x = get_cormorant_features(one_hot, z, 2, max_scale)

#     graph.z = z.unsqueeze(-1)
    graph.edge_index = None
    return graph


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def calc_mean_mad(dataset) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)

#     biggest_dist = 0.0
#     for graph in dataset:
#         norms = torch.norm(graph.pos, dim=1)
#         max_norm = torch.max(norms).item()
#         if biggest_dist < max_norm:
#             biggest_dist = max_norm
#
#
#     print(biggest_dist)
#     quit()

    return mean, mad