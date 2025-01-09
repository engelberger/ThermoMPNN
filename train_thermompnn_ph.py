import sys
import wandb
import os

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
        assert len(batch) == 1
        mut_pdb, mutations = batch[0]
        pred, _ = self(mut_pdb, mutations)

        ph_mses = []
        ph_preds = []
        ph_targets = []

        for mut, out in zip(mutations, pred):
            if mut.pH is not None:  # Assuming pH is added to mutation objects
                ph_mses.append(F.mse_loss(out["pH"], mut.pH))
                for metric in self.metrics[f"{prefix}_metrics"]["pH"].values():
                    metric.update(out["pH"], mut.pH)
                ph_preds.append(out["pH"])
                ph_targets.append(mut.pH)

        loss = 0.0 if len(ph_mses) == 0 else torch.stack(ph_mses).mean()
        on_step = False
        on_epoch = not on_step

        output = "pH"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(
                f"{prefix}_{output}_{name}",
                metric,
                prog_bar=True,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=len(batch),
            )

        # Return predictions and targets for monitoring
        if len(ph_preds) > 0:
            return {
                "loss": loss,
                "predictions": torch.stack(ph_preds),
                "targets": torch.stack(ph_targets),
            }
        return {"loss": loss} if loss != 0.0 else None

    def training_step(self, batch, batch_idx):
        outputs = self.shared_eval(batch, batch_idx, "train")
        return outputs["loss"] if outputs is not None else None

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, "test")

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


def train(cfg):
    """Main training function"""
    # Set up datasets
    train_workers = cfg.platform.train_workers if "train_workers" in cfg.platform else 4
    val_workers = cfg.platform.val_workers if "val_workers" in cfg.platform else 4

    # Load pH datasets
    train_dataset = PHDataset(cfg, "train")
    val_dataset = PHDataset(cfg, "val")

    train_loader = DataLoader(
        train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=lambda x: x, num_workers=val_workers
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
    )
    trainer.fit(model_pl, train_loader, val_loader)

    if "two_stage" in cfg.training:  # sequential combo training
        if cfg.training.two_stage:
            print("Two-stage Training Enabled")
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = PHDataset(cfg, "train")
            val_dataset = PHDataset(cfg, "val")
            train_loader = DataLoader(
                train_dataset,
                collate_fn=lambda x: x,
                shuffle=True,
                num_workers=train_workers,
            )
            val_loader = DataLoader(
                val_dataset, collate_fn=lambda x: x, num_workers=val_workers
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
