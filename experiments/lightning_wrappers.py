import glob

import pytorch_lightning as pl

# torch
import torch
import torchmetrics
import wandb
from ml_collections import config_dict

import plotly.graph_objects as go
# project
import experiments
import src
import src.nn as src_nn
import numpy as np
import torch.nn.functional as F
import torchvision
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt


class LightningWrapperBase(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
    ):
        super().__init__()
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        # Placeholders for logging of best train & validation values
        self.num_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1

        self.time_inference = cfg.track_inference_time
        if self.time_inference:
            self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def forward(self, x, time_inference=False):
        if time_inference:
            self.start_timer()


        out = self.network(x)

        if time_inference:
            self.end_timer()

        return out

    def start_timer(self):
        self.starter.record()

    def end_timer(self):
        self.ender.record()
        torch.cuda.synchronize()
        val_time = self.starter.elapsed_time(self.ender)
        self.log(
            "val/memory_allocated",
            torch.cuda.memory_allocated() /1024/1024/1024
        )
        self.log(
            "val/inference_time",
            val_time
        )

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        optimizer = experiments.construct_optimizer(model=self, optim_cfg=self.optim_cfg)
        scheduler = experiments.construct_scheduler(
            optimizer=optimizer, scheduler_cfg=self.scheduler_cfg
        )
        # Construct output dictionary
        output_dict = {"optimizer": optimizer}
        if scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"
            # TODO(dwromero): ReduceLROnPlateau with warmup
            if isinstance(scheduler, src_nn.schedulers.ChainedScheduler) and isinstance(
                scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        return output_dict

    def on_train_start(self):
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.num_params == -1:
                with torch.no_grad():
                    # Log parameters
                    num_params = src.utils.num_params(self.network)
                    self.logger.experiment.summary["num_params"] = num_params
                    self.num_params = num_params
                    # Log source code files
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths of all source code files
                    paths = glob.glob("**/*.py", recursive=True)
                    paths += glob.glob("**/*.yaml", recursive=True)
                    # Filter paths
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(path, name=path)
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)
        # Metric computers
        if cfg.net.out_channels == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=cfg.net.out_channels, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.test_f1 = torchmetrics.F1Score(task=task, num_classes=cfg.net.out_channels, top_k=1)
        self.loss_metric = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.batch_size = cfg.train.batch_size

        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0

        # Compute predictions
        self.get_predictions = lambda logits: torch.argmax(logits, 1)

    def _step(self, batch, accuracy_calculator, time_inference=False, f1_calculator=None):
        x = batch

        logits = self(x, time_inference)

        # Predictions
        predictions = self.get_predictions(logits)
        # Calculate accuracy and loss
        labels = batch.y

        accuracy_calculator(predictions, labels)

        if f1_calculator is not None:
            f1_calculator(predictions, labels)

        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.train_acc, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.training_step_outputs.append(logits.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.val_acc, time_inference=self.time_inference, f1_calculator=self.val_f1)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "val/F1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size
        )
        self.validation_step_outputs.append(logits)
        return logits  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, self.test_acc, time_inference=False, f1_calculator=self.test_f1)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "test/F1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size
        )

    def on_train_epoch_end(self):
        flattened_logits = torch.cat(self.training_step_outputs)
        flattened_logits = torch.flatten(flattened_logits)
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        self.training_step_outputs.clear()
        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            self.logger.experiment.log(
                {
                    "train/best_acc": self.best_train_acc,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(torch.cat(self.validation_step_outputs))

        self.logger.experiment.log(
            {
                "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                "global_step": self.global_step,
            }
        )
        self.validation_step_outputs.clear()
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_acc,
                    "global_step": self.global_step,
                }
            )


class SegmentationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)
        # Metric computers
        if cfg.net.out_channels == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.test_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)

        self.time_inference = cfg.track_inference_time

        self.num_classes = cfg.net.out_channels

        self.loss_metric = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.batch_size = cfg.train.batch_size

        self.best_train_mAcc = 0.0
        self.best_val_mAcc = 0.0
        self.best_val_ins_miou = 0.0
        self.best_val_cat_miou = 0.0

        assert cfg.dataset.name == "ShapeNet", "currently hardcoded for shapenet"

        self.global_train_iou = None
        self.global_val_iou = None
        self.category_to_seg_classes = {
            "Earphone": [16, 17, 18],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Rocket": [41, 42, 43],
            "Car": [8, 9, 10, 11],
            "Laptop": [28, 29],
            "Cap": [6, 7],
            "Skateboard": [44, 45, 46],
            "Mug": [36, 37],
            "Guitar": [19, 20, 21],
            "Bag": [4, 5],
            "Lamp": [24, 25, 26, 27],
            "Table": [47, 48, 49],
            "Airplane": [0, 1, 2, 3],
            "Pistol": [38, 39, 40],
            "Chair": [12, 13, 14, 15],
            "Knife": [22, 23],
        }
        # inverse mapping
        self.seg_class_to_category = {}
        for cat in self.category_to_seg_classes.keys():
            for cls in self.category_to_seg_classes[cat]:
                self.seg_class_to_category[cls] = cat

        # Compute predictions
        self.get_predictions = lambda logits: torch.argmax(logits, 1)

    def _step(self, batch, time_inference=False, acc_calculator=None):
        x = batch

        logits = self(x, time_inference)

        # Predictions
        predictions = self.get_predictions(logits)

        # Calculate accuracy and loss

        labels = batch.y
        acc_calculator(predictions, labels)

        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss= self._step(batch, time_inference=False, acc_calculator=self.train_mAcc)

        # Log and return loss (Required in training step)
        self.log("train/loss",
                 loss.item(),
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )

        self.log("train/mAcc",
                 self.train_mAcc,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )
        self.training_step_outputs.append(logits.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, time_inference=self.time_inference, acc_calculator=self.val_mAcc)

        ious = self.compute_shape_ious(logits.reshape(self.batch_size, -1, self.num_classes), batch.y.reshape(self.batch_size, -1))
        self.validation_step_outputs.append(ious)

        # Log and return loss (Required in training step)
        self.log("val/loss",
                 loss.item(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )


        self.log("val/mAcc",
                 self.val_mAcc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size
                )

        return ious  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, time_inference=False, acc_calculator=self.test_mAcc)
        # Log and return loss (Required in training step)
        self.log("test/loss",
                 loss.item(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )

        self.log("test/mAcc",
                 self.test_mAcc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size
                )


    def on_train_epoch_end(self):
        flattened_logits = torch.cat(self.training_step_outputs)
        flattened_logits = torch.flatten(flattened_logits)
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        self.training_step_outputs.clear()

        train_mAcc = self.trainer.callback_metrics["train/mAcc_epoch"]
        if train_mAcc > self.best_train_mAcc:
            self.best_train_mAcc = train_mAcc.item()
            self.logger.experiment.log(
                {
                    "train/best_mAcc": self.best_train_mAcc,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.

        all_shape_mious, cat_mious = self.get_mious(self.validation_step_outputs)

        self.log("val/ins_miou", all_shape_mious.mean())
        self.log("val/cat_miou", torch.stack(list(cat_mious.values())).mean())
        for cat in sorted(cat_mious.keys()):
            self.log(f"val/cat_miou_{cat}", cat_mious[cat])

        self.validation_step_outputs.clear()

        val_mAcc = self.trainer.callback_metrics["val/mAcc"]
        if val_mAcc > self.best_val_mAcc:
            self.best_val_mAcc = val_mAcc.item()
            self.logger.experiment.log(
                {
                    "val/best_mAcc": self.best_val_mAcc,
                    "global_step": self.global_step,
                }
            )

        val_ins_miou = self.trainer.callback_metrics["val/ins_miou"]
        if val_ins_miou > self.best_val_ins_miou:
            self.best_val_ins_miou = val_ins_miou.item()
            self.logger.experiment.log(
                {
                    "val/best_ins_miou": self.best_val_ins_miou,
                    "global_step": self.global_step,
                }
            )

        val_cat_miou = self.trainer.callback_metrics["val/cat_miou"]
        if val_cat_miou > self.best_val_cat_miou:
            self.best_val_cat_miou = val_cat_miou.item()
            self.logger.experiment.log(
                {
                    "val/best_cat_miou": self.best_val_cat_miou,
                    "global_step": self.global_step,
                }
            )


    # Metric calculation taken and adapted from https://github.com/kabouzeid/point2vec
    def compute_shape_ious(self, log_probabilities, seg_labels):
        # log_probablities: (B, N, 50) \in -inf..<0
        # seg_labels:       (B, N) \in 0..<50
        # returns           { cat: (S, P) }


        shape_ious = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }

        for i in range(log_probabilities.shape[0]):
            cat = self.seg_class_to_category[seg_labels[i, 0].item()]
            seg_classes = self.category_to_seg_classes[cat]
            seg_preds = (
                torch.argmax(
                    log_probabilities[i, :, self.category_to_seg_classes[cat]], dim=1
                )
                + seg_classes[0]
            )  # (N,)

            seg_class_iou = torch.empty(len(seg_classes))
            for c in seg_classes:
                if ((seg_labels[i] == c).sum() == 0) and (
                    (seg_preds == c).sum() == 0
                ):  # part is not present, no prediction as well
                    seg_class_iou[c - seg_classes[0]] = 1.0
                else:
                    intersection = ((seg_labels[i] == c) & (seg_preds == c)).sum()
                    union = ((seg_labels[i] == c) | (seg_preds == c)).sum()
                    seg_class_iou[c - seg_classes[0]] = intersection / union
            shape_ious[cat].append(seg_class_iou.mean())

        return shape_ious

    def get_mious(self, outputs):
        shape_mious = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }
        for d in outputs:
            for k, v in d.items():
                shape_mious[k] = shape_mious[k] + v

        all_shape_mious = torch.stack(
            [miou for mious in shape_mious.values() for miou in mious]
        )
        cat_mious = {
            k: torch.stack(v).mean() for k, v in shape_mious.items() if len(v) > 0
        }
        return all_shape_mious, cat_mious



class QM9Wrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        mean,
        mad,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)

        self.mean = mean
        self.mad = mad

        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.val_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

        self.loss_metric = torch.nn.L1Loss(reduction='sum')

        self.equi_loss = torch.nn.L1Loss(reduction='sum')

        self.batch_size = cfg.train.batch_size

        # Placeholders for logging of best train & validation values
        self.best_train_loss = np.inf
        self.best_val_mae = np.inf


    def _step(self, batch, loss_calculator, mae_calculator, time_inference=False, visualize=False):
        x = batch

        if visualize:
            logger = self.logger.experiment
        else:
            logger = None

        pred = self.network(x, logger=logger)

        loss = loss_calculator(pred, (batch.y - self.mean) / self.mad)

        mae_calculator(pred * self.mad + self.mean, batch.y)

        # Return predictions and loss
        return loss

    def training_step(self, batch, batch_idx):
        # Perform step


        loss = self._step(batch, self.loss_metric, self.train_metric, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "train/mae",
            self.train_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        visualize = False
        if batch_idx == 0:
#             self.equivariance_loss(batch[0])
            out, out_pos, _, _ = self.network.grid_representation(batch)
#             self.plot_grid(out)
            visualize = True
#         out, pos, _, _ = self.network.grid_representation(batch)

        loss = self._step(batch, self.loss_metric, self.val_metric, time_inference=False, visualize=visualize)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "val/mae",
            self.val_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        # Perform step
        loss = self._step(batch, self.loss_metric, self.test_metric, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "test/mae",
            self.test_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )


        return loss

    def on_train_epoch_end(self):
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Log best accuracy
        val_mae = self.trainer.callback_metrics["val/mae"]
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae.item()
            self.logger.experiment.log(
                {
                    "val/best_mae": self.best_val_mae,
                    "global_step": self.global_step,
                }
            )


    def equivariance_loss(self, graph):


        angle = np.random.uniform(0, 2*np.pi, 3)
        def get_rotation_matrices(angle_x, angle_y, angle_z, device):
            R_z = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, np.cos(angle_z), -np.sin(angle_z)],
                            [0.0, np.sin(angle_z),  np.cos(angle_z)]], dtype=torch.float32, device=graph.pos.device)

            R_y = torch.tensor([[np.cos(angle_y), 0.0, np.sin(angle_y)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angle_y), 0.0, np.cos(angle_y)]], dtype=torch.float32, device=graph.pos.device)

            R_x = torch.tensor([[np.cos(angle_x), -np.sin(angle_x), 0.0],
                                [np.sin(angle_x),  np.cos(angle_x), 0.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float32, device=graph.pos.device)

            return R_x, R_y, R_z

        angle = 0.5 * np.pi
        R_x, R_y, R_z = get_rotation_matrices(-angle, 0, 0, graph.pos.device)

        rot_pos = graph.pos @ R_x @ R_y @ R_z


        graph_rot = graph.clone()


        graph_rot.pos = rot_pos


        out, out_pos, _, _ = self.network.grid_representation(graph)
#         self.plot_grid(out[:,:1,:,:,:], name="f(x)")

        out_rot, out_pos_rot, _, _ = self.network.grid_representation(graph_rot)
        print(out_rot.shape)
        print(out_pos_rot.shape)

        self.plot_grid_3d(out_pos_rot, out_rot[0, 0, :,:,:].view(-1, 1))

        self.plot_grid(out_rot[:,:1,:,:,:], name="f(t(x))")

        quit()

#         R_x, R_y, R_z = get_rotation_matrices(0, 0, -angle, graph.pos.device)
#         inv_rot_pos = out_pos_rot @ R_x @ R_y @ R_z

#         delta
#         grid_res = self.network.grid_representation.grid_resolution
#         out = torch.zeros(out_rot.shape, device=out_rot.device)
#         out[0, :, grid_res//2-1, grid_res//2-1, grid_res//2-1] = 1.0
#         self.plot_grid(out, name="f(x)")


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

        out_rot = self.rotate_3d(out_rot, -angle * 180 / np.pi)

        self.plot_grid(out_rot, name="t-1(f(t(x)))")
        quit()
#
#         eq_error = ((out - out_rot) ** 2).sum() / (out_rot ** 2).sum()
#
#         self.logger.experiment.log(
#                 {
#                     "val/equivariance_loss": eq_error,
#                     "global_step": self.global_step,
#                 }
#             )
    def rotate_3d(self, imgs, angle, axis="z"):
        height = imgs.shape[2]

        img_list = []
        for h in range(height):
            out = torchvision.transforms.functional.rotate(imgs[:,:1, h, :,:], angle=angle)
            img_list.append(out.unsqueeze(2))


        return torch.cat(img_list, dim=2)




    def plot_object(self, f, pos):

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

    def plot_grid_3d(self, positions, features):
        positions = positions.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        xdata, ydata, zdata = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c=features, s=20)
        fig.show()

    def plot_grid(self, grid_rep, name="post_conv_repr", channels=3):
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

            self.logger.experiment.log({name:wandb.Image(fig)})
#             plt.close()

        grid_rep = grid_rep[0, :channels, :, :, :].detach().cpu().numpy()

        d_slice = grid_rep.shape[1]
        ims = []
        for d in range(d_slice):
            imslice = grid_rep[:, d, :, :].transpose(1, 2, 0)
            ims.append(imslice)

        tile_images(ims)

        plt.show()

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



class MD17Wrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)
        # Metric computers
        self.force_weight = 1000
        self.shift = -17591926.0
        self.scale = 2351.177978515625

        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_val_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.force_train_metric = torchmetrics.MeanAbsoluteError()
        self.force_val_metric = torchmetrics.MeanAbsoluteError()
        self.force_test_metric = torchmetrics.MeanAbsoluteError()
        self.loss_metric = torch.nn.MSELoss()

        self.batch_size = cfg.train.batch_size

        # Placeholders for logging of best train & validation values
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf

    def pred_energy_and_force(self, graph, logger=None):
        graph.pos = torch.autograd.Variable(graph.pos, requires_grad=True)
        pred_energy = self.network(graph, logger=logger)

        sign = -1.0
        pred_force = (
            sign
            * torch.autograd.grad(
                pred_energy,
                graph.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=True,
                retain_graph=True,
            )[0]
        )
        return pred_energy.squeeze(-1), pred_force

    def energy_and_force_loss(self, graph, energy, force):

        loss = F.mse_loss(energy, (graph.energy - self.shift) / self.scale)
        loss = loss + self.force_weight * F.mse_loss(force, graph.force / self.scale)
        return loss

    def equivariance_loss(self, graph):


        angle = np.random.uniform(0, 2*np.pi, 3)
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, np.cos(angle[0]), -np.sin(angle[0])],
                            [0.0, np.sin(angle[0]),  np.cos(angle[0])]], dtype=torch.float32, device=graph.pos.device)

        R_y = torch.tensor([[np.cos(angle[1]), 0.0, np.sin(angle[1])],
                            [0.0, 1.0, 0.0],
                            [-np.sin(angle[1]), 0.0, np.cos(angle[1])]], dtype=torch.float32, device=graph.pos.device)

        R_z = torch.tensor([[np.cos(angle[2]), -np.sin(angle[2]), 0.0],
                            [np.sin(angle[2]),  np.cos(angle[2]), 0.0],
                            [0.0, 0.0, 1.0]], dtype=torch.float32, device=graph.pos.device)

        rot_pos = graph.pos @ R_x @ R_y @ R_z

        graph_rot = graph.clone()

        graph_rot.pos = rot_pos
#
#         self.plot_object(graph.x, graph.pos)
#         self.plot_object(graph.x, rot_pos)


        out, out_pos, _, _ = self.network.grid_representation(graph)
#         self.plot_grid(out)
#         out_rot, out_pos_rot, _, _ = self.network.grid_representation(graph_rot)
#
        out_pos = out_pos @ R_x @ R_y @ R_z
        grid_res = self.network.grid_representation.grid_resolution
        out_pos = out_pos.reshape(1,
                                   grid_res,
                                   grid_res,
                                   grid_res,
                                   -1).permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format)

        out = torch.nn.functional.grid_sample(out, out_pos)


        eq_error = ((out - out_rot) ** 2).sum() / (out_rot ** 2).sum()

        self.logger.experiment.log(
                {
                    "val/equivariance_loss": eq_error,
                    "global_step": self.global_step,
                }
            )


    def _step(self, batch, loss_calculator, energy_metric, force_metric, time_inference=False, visualize=False):
        x = batch

        if visualize:
            logger = self.logger.experiment
        else:
            logger = None

#         pred = self.network(x, logger=logger)

        energy, force = self.pred_energy_and_force(x, logger=logger)

        loss = loss_calculator(x, energy, force)

        energy_metric(energy * self.scale + self.shift, x.energy)
        force_metric(force * self.scale, x.force)

        # Return predictions and loss
        return loss

    def training_step(self, batch, batch_idx):
        # Perform step

        loss = self._step(batch, self.energy_and_force_loss, self.energy_train_metric, self.force_train_metric, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "train/energy_mae",
            self.energy_train_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "train/force_mae",
            self.force_train_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        return loss

    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx):
        # Perform step
        visualize = False
        if batch_idx == 0:
#             self.equivariance_loss(batch[0])
            if not "MPNN" in str(type(self.network)):
                out, out_pos, _, _ = self.network.grid_representation(batch)
                self.plot_grid(out)
                visualize = True
#         out, pos, _, _ = self.network.grid_representation(batch)

        loss = self._step(batch, self.energy_and_force_loss, self.energy_val_metric, self.force_val_metric, time_inference=False, visualize=visualize)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "val/energy_mae",
            self.energy_val_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "val/force_mae",
            self.force_val_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        return loss

    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        # Perform step
        loss = self._step(batch, self.energy_and_force_loss, self.energy_test_metric, self.force_test_metric, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "test/energy_mae",
            self.energy_test_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "test/force_mae",
            self.force_test_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        return loss

    def on_train_epoch_end(self):
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Log best accuracy
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )

    def plot_object(self, f, pos):

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

    def plot_grid(self, grid_rep):
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

            self.logger.experiment.log({"grid_repr_post_conv":wandb.Image(fig)})
#             plt.close()

        grid_rep = grid_rep[0, :3, :, :, :].detach().cpu().numpy()

        d_slice = grid_rep.shape[1]
        ims = []
        for d in range(d_slice):
            imslice = grid_rep[:, d, :, :].transpose(1, 2, 0)
            ims.append(imslice)

        tile_images(ims)