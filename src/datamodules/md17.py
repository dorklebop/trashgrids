# torch
import pytorch_lightning as pl
import torch
import torch_geometric.datasets as tg_datasets
from torch.utils.data import DataLoader, random_split
from torch_geometric import loader as tg_loader

import torch_geometric.transforms as tg_transforms

from torchvision import datasets, transforms


class MD17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        **kwargs,
    ):
        super().__init__()

        num_classes = cfg.params.num_classes
        use_positions = cfg.params.use_positions
        use_normals = cfg.params.use_normals
        data_dir = cfg.data_dir
        num_samples = cfg.params.num_samples

        # Save parameters to self
        self.data_dir = data_dir + f"/md17_aspirin"
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.use_positions = use_positions
        self.use_normals = use_normals
        self.num_workers = num_workers
        self.pin_memory = pin_memory

#         self.min_pos = -4.718296527862549
#         self.max_pos = 4.718296527862549

        self.min_pos = -5
        self.max_pos = 5

        self.data_dim = 3
        self.input_channels = 10
        self.output_channels = self.num_classes

        transform = [tg_transforms.Center(),
                     ztox(),
                     Kcal2meV()]

        if cfg.md17.add_edges:
            transform += [tg_transforms.RadiusGraph(4.0)]

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


    def prepare_data(self):
        # download
        tg_datasets.MD17(
            root=self.data_dir,
            name="aspirin CCSD",
            train=True,
            pre_transform=self.pre_transform)
        tg_datasets.MD17(
            root=self.data_dir,
            name="aspirin CCSD",
            train=False,
            pre_transform=self.pre_transform)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            md17_train = tg_datasets.MD17(
                root=self.data_dir,
                name="aspirin CCSD",
                train=True,
                transform=self.train_transform
            )

            splits = [950, 50]
            self.train_dataset, self.val_dataset = random_split(
                md17_train,
                splits,
                generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
            )

        if stage == "test" or stage is None:
            md17_test = tg_datasets.MD17(
                root=self.data_dir,
                name="aspirin CCSD",
                train=False,
                transform=self.test_transform,
            )

            self.test_dataset = md17_test

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




# Let's load the datasets.
class Kcal2meV:
    def __init__(self):
        # Kcal/mol to meV
        self.conversion = 43.3634

    def __call__(self, graph):
        graph.energy = graph.energy * self.conversion
        graph.force = graph.force * self.conversion
        return graph


class ztox:
    def __init__(self):
        pass
    def __call__(self, graph):
        graph.x = graph.z
        return graph

