import sys
import wandb
import os
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from transfer_model_ph import TransferModelPH
from datasets import PHDataset
from analysis.training_monitor import PHTrainingMonitor


def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


class TransferModelPHPL(pl.LightningModule):
    """Class managing training loop with pytorch lightning for pH prediction"""

    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModelPH(cfg)

        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = (
            cfg.training.mpnn_learn_rate if "mpnn_learn_rate" in cfg.training else None
        )
        self.lr_schedule = (
            cfg.training.lr_schedule if "lr_schedule" in cfg.training else False
        )

        # set up metrics dictionary
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "pH"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        """Shared evaluation step for training, validation, and testing"""
        try:
            # Extract mut_pdb and mutations
            if isinstance(batch, list) and len(batch) > 0:
                if isinstance(batch[0], list) and len(batch[0]) == 2:
                    mut_pdb, mutations = batch[0]
                elif len(batch) == 2:
                    mut_pdb, mutations = batch
                else:
                    return {"loss": torch.tensor(0.0, requires_grad=True)}
            else:
                return {"loss": torch.tensor(0.0, requires_grad=True)}

            # Forward pass
            pred, _ = self(mut_pdb, mutations)

            # Calculate metrics
            ph_mses = []
            ph_preds = []
            ph_targets = []

            for mut, out in zip(mutations, pred):
                if mut.pH is not None:  # Only process if pH value exists
                    ph_mses.append(F.mse_loss(out["pH"], mut.pH))
                    for metric in self.metrics[f"{prefix}_metrics"]["pH"].values():
                        metric.update(out["pH"], mut.pH)
                    ph_preds.append(out["pH"])
                    ph_targets.append(mut.pH)

            # Handle empty batch case
            if len(ph_mses) == 0:
                return {"loss": torch.tensor(0.0, requires_grad=True)}

            # Calculate loss and log metrics
            loss = torch.stack(ph_mses).mean()

            # Log metrics
            output = "pH"
            for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
                try:
                    metric_val = metric.compute()
                    self.log(
                        f"{prefix}_{output}_{name}",
                        metric_val,
                        prog_bar=True,
                        on_step=False,
                        on_epoch=True,
                        batch_size=len(ph_preds),
                    )
                except ValueError:
                    continue

            # Return predictions and targets for monitoring
            return {
                "loss": loss,
                "predictions": torch.stack(ph_preds) if ph_preds else None,
                "targets": torch.stack(ph_targets) if ph_targets else None,
            }
        except Exception as e:
            # Only show debug output on actual errors
            print(f"\nError in shared_eval: {str(e)}")
            print(f"Batch structure: {type(batch)}")
            if isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
                if len(batch) > 0:
                    print(f"First element type: {type(batch[0])}")
                    if isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0:
                        print(f"First element contents: {[type(x) for x in batch[0]]}")
            return {"loss": torch.tensor(0.0, requires_grad=True)}

    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self.shared_eval(batch, batch_idx, "train")
        if outputs is None:
            return None
        return outputs["loss"] if "loss" in outputs else None

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self.shared_eval(batch, batch_idx, "val")
        if outputs is None:
            return None
        return outputs

    def test_step(self, batch, batch_idx):
        """Test step"""
        outputs = self.shared_eval(batch, batch_idx, "test")
        if outputs is None:
            return None
        return outputs

    def configure_optimizers(self):
        if self.stage == 2:  # for second stage, drop LR by factor of 10
            self.learn_rate /= 10.0
            print("New second-stage learning rate: ", self.learn_rate)

        if not self.hparams.cfg.model.freeze_weights:  # fully unfrozen ProteinMPNN
            param_list = [
                {
                    "params": self.model.prot_mpnn.parameters(),
                    "lr": self.mpnn_learn_rate,
                }
            ]
        else:  # fully frozen MPNN
            param_list = []

        if self.model.lightattn:  # adding light attention parameters
            if self.stage == 2:
                param_list.append(
                    {"params": self.model.light_attention.parameters(), "lr": 0.0}
                )
            else:
                param_list.append({"params": self.model.light_attention.parameters()})

        # Modified for pH prediction layers
        mlp_params = [{"params": self.model.ph_out.parameters()}]

        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule:  # enable additional lr scheduler conditioned on val pH mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt, verbose=True, mode="min", factor=0.5
            )
            return {"optimizer": opt, "lr_scheduler": lr_sched, "monitor": "val_pH_mse"}
        else:
            return opt


# Worker initialization function
def worker_init_fn(worker_id):
    # Set numpy seed for this worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    # Ensure matplotlib doesn't try to use GUI backend in workers
    matplotlib.use("Agg")
    # Disable parallel processing in workers to prevent deadlocks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def collate_fn(batch):
    """Custom collate function to handle batches of mutation data"""
    if len(batch) == 0:
        return []
    return batch[0]  # Return first item since we're already batching in the dataset


def train(cfg):
    """Main training function"""
    # Set DataLoader worker settings
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Set number of workers based on CPU count and config
    n_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0

    # Load pH datasets
    train_dataset = PHDataset(cfg, "train")
    val_dataset = PHDataset(cfg, "val")

    # Create data loaders with modified settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=n_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if n_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if n_workers > 0 else None,
        collate_fn=collate_fn,  # Use custom collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=n_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if n_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if n_workers > 0 else None,
        collate_fn=collate_fn,  # Use custom collate function
    )

    model_pl = TransferModelPHPL(cfg)
    model_pl.stage = 1

    # Set up callbacks
    callbacks = []

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Checkpoint callback
    filename = cfg.name + "_{epoch:02d}_{val_pH_spearman:.2f}"
    monitor = "val_pH_spearman"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        dirpath="checkpoints",
        filename=filename,
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Training monitor callback
    monitor_callback = PHTrainingMonitor(
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        plot_every_n_epochs=cfg.training.get("plot_every_n_epochs", 1),
        gradient_norm_logging=cfg.training.get("gradient_norm_logging", True),
        activation_histograms=cfg.training.get("activation_histograms", True),
        save_dir=cfg.training.get("monitor_save_dir", "training_plots"),
    )
    callbacks.append(monitor_callback)

    # Set up logger with fixed settings
    if "project" in cfg:
        os.environ["WANDB_SILENT"] = "true"  # Reduce wandb logging noise
        logger = WandbLogger(
            project=cfg.project,
            name="PH_Prediction",
            log_model=False,  # Disable automatic model logging
            save_dir="wandb",
        )
    else:
        logger = None
    max_ep = cfg.training.epochs if "epochs" in cfg.training else 100

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        max_epochs=max_ep,
        accelerator=cfg.platform.accel,
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=None,  # No gradient clipping
        deterministic=False,  # Allow non-deterministic operations for speed
        precision=cfg.training.precision,  # Use precision from config
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,  # Gradient accumulation from config
    )

    # Print dataset sizes
    print(
        f"\nTraining on {len(train_dataset)} examples with batch size {cfg.training.batch_size}"
    )
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Validation set size: {len(val_dataset)}\n")

    trainer.fit(model_pl, train_loader, val_loader)

    if "two_stage" in cfg.training:  # sequential combo training
        if cfg.training.two_stage:
            print("Two-stage Training Enabled")
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = PHDataset(cfg, "train")
            val_dataset = PHDataset(cfg, "val")

            # Use same DataLoader configuration as first stage
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=n_workers,
                worker_init_fn=worker_init_fn,
                persistent_workers=True if n_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=2 if n_workers > 0 else None,
                collate_fn=collate_fn,  # Use custom collate function
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=n_workers,
                worker_init_fn=worker_init_fn,
                persistent_workers=True if n_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=2 if n_workers > 0 else None,
                collate_fn=collate_fn,  # Use custom collate function
            )

            model_pl.stage = 2
            # re-start training with a new trainer
            trainer = pl.Trainer(
                callbacks=callbacks,  # Reuse the same callbacks
                logger=logger,
                log_every_n_steps=10,
                max_epochs=max_ep * 2,
                accelerator=cfg.platform.accel,
                devices=1,
            )
            trainer.fit(
                model_pl,
                train_loader,
                val_loader,
                ckpt_path=checkpoint_callback.best_model_path,
            )


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) == 1:
        yaml = "config.yaml"
    else:
        yaml = sys.argv[1]

    cfg = OmegaConf.load(yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    train(cfg)
