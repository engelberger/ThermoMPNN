import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import Callback
import torch
import wandb


class PHTrainingMonitor(Callback):
    """Monitor training progress for pH prediction."""

    def __init__(
        self,
        log_every_n_steps: int = 50,
        plot_every_n_epochs: int = 1,
        gradient_norm_logging: bool = True,
        activation_histograms: bool = True,
        save_dir: str = "training_plots",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.plot_every_n_epochs = plot_every_n_epochs
        self.gradient_norm_logging = gradient_norm_logging
        self.activation_histograms = activation_histograms
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize storage
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.grad_norms = []
        self.ph_targets = []
        self.ph_predictions = []
        self.batch_idx = 0

        # For epoch-level tracking
        self.current_train_loss = []
        self.current_val_loss = []
        self.current_test_loss = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_test_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics."""
        if outputs is None:
            return

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.current_train_loss.append(loss.item())

        if batch_idx % self.log_every_n_steps == 0:
            if self.gradient_norm_logging:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    pl_module.parameters(), float("inf")
                ).item()
                self.grad_norms.append(grad_norm)
                if trainer.logger:
                    trainer.logger.experiment.log(
                        {"gradient_norm": grad_norm, "global_step": trainer.global_step}
                    )

            self.train_losses.append(loss.item())
            self.batch_idx = batch_idx

            # Log to wandb
            if trainer.logger:
                trainer.logger.experiment.log(
                    {"train_loss": loss.item(), "global_step": trainer.global_step}
                )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training epoch metrics."""
        if self.current_train_loss:
            epoch_loss = np.mean(self.current_train_loss)
            self.epoch_train_losses.append(epoch_loss)
            if trainer.logger:
                trainer.logger.experiment.log(
                    {"epoch_train_loss": epoch_loss, "epoch": trainer.current_epoch}
                )
        self.current_train_loss = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect predictions and validation loss."""
        if outputs is None:
            return

        # Store validation loss
        if "loss" in outputs:
            self.current_val_loss.append(outputs["loss"].item())

        # Store predictions if available
        if "predictions" in outputs and "targets" in outputs:
            predictions = outputs["predictions"]
            targets = outputs["targets"]
            self.ph_targets.extend(targets.cpu().numpy())
            self.ph_predictions.extend(predictions.cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate validation plots and log metrics."""
        if self.current_val_loss:
            epoch_loss = np.mean(self.current_val_loss)
            self.epoch_val_losses.append(epoch_loss)
            if trainer.logger:
                trainer.logger.experiment.log(
                    {"epoch_val_loss": epoch_loss, "epoch": trainer.current_epoch}
                )

        if trainer.current_epoch % self.plot_every_n_epochs == 0:
            self._plot_predictions(trainer.current_epoch)
            self._analyze_ph_ranges(trainer.current_epoch)
            self._plot_loss_curves(trainer.current_epoch)
            self._plot_wandb_curves(trainer)

            # Log plots to wandb
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                for plot_name in ["predictions", "error_by_range", "training_curves"]:
                    plot_path = os.path.join(
                        self.save_dir, f"{plot_name}_epoch_{trainer.current_epoch}.png"
                    )
                    if os.path.exists(plot_path):
                        trainer.logger.experiment.log(
                            {
                                f"{plot_name}": wandb.Image(plot_path),
                                "epoch": trainer.current_epoch,
                            }
                        )

        # Clear storage for next epoch
        self.ph_targets = []
        self.ph_predictions = []
        self.current_val_loss = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect test metrics."""
        if outputs is None or "loss" not in outputs:
            return
        self.current_test_loss.append(outputs["loss"].item())

    def on_test_epoch_end(self, trainer, pl_module):
        """Log test epoch metrics."""
        if self.current_test_loss:
            epoch_loss = np.mean(self.current_test_loss)
            self.epoch_test_losses.append(epoch_loss)
            if trainer.logger:
                trainer.logger.experiment.log(
                    {"epoch_test_loss": epoch_loss, "epoch": trainer.current_epoch}
                )
        self.current_test_loss = []

    def _plot_wandb_curves(self, trainer):
        """Create and log loss curves to wandb."""
        if not trainer.logger:
            return

        # Skip if we don't have any data yet
        if (
            not self.epoch_train_losses
            and not self.epoch_val_losses
            and not self.epoch_test_losses
        ):
            return

        # Create figure
        plt.figure(figsize=(10, 6))

        # Get the current epoch range
        max_epochs = max(
            len(self.epoch_train_losses),
            len(self.epoch_val_losses),
            len(self.epoch_test_losses),
        )
        epochs = list(range(max_epochs))

        # Plot available losses
        if self.epoch_train_losses:
            plt.plot(
                epochs[: len(self.epoch_train_losses)],
                self.epoch_train_losses,
                label="Train",
                marker="o",
            )

        if self.epoch_val_losses:
            plt.plot(
                epochs[: len(self.epoch_val_losses)],
                self.epoch_val_losses,
                label="Validation",
                marker="s",
            )

        if self.epoch_test_losses:
            plt.plot(
                epochs[: len(self.epoch_test_losses)],
                self.epoch_test_losses,
                label="Test",
                marker="^",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        # Log to wandb
        trainer.logger.experiment.log(
            {"loss_curves": wandb.Image(plt), "epoch": trainer.current_epoch}
        )
        plt.close()

    def _plot_predictions(self, epoch: int):
        """Create scatter plot of predictions vs targets."""
        if not self.ph_targets or not self.ph_predictions:
            return

        plt.figure(figsize=(10, 10))

        # Convert to flat arrays for plotting
        targets = np.array(self.ph_targets).flatten()
        predictions = np.array(self.ph_predictions).flatten()

        # Create DataFrame for seaborn
        df = pd.DataFrame({"True pH": targets, "Predicted pH": predictions})

        # Create scatter plot
        sns.scatterplot(data=df, x="True pH", y="Predicted pH", alpha=0.5)
        plt.plot([0, 14], [0, 14], "r--")  # Add diagonal line

        plt.title(f"pH Predictions (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"predictions_epoch_{epoch}.png"))
        plt.close()

    def _analyze_ph_ranges(self, epoch: int):
        """Analyze performance across pH ranges."""
        if not self.ph_targets or not self.ph_predictions:
            return

        # Convert to flat arrays
        targets = np.array(self.ph_targets).flatten()
        predictions = np.array(self.ph_predictions).flatten()
        errors = np.abs(predictions - targets)

        # Create DataFrame with pH ranges
        df = pd.DataFrame(
            {"True pH": targets, "Predicted pH": predictions, "Error": errors}
        )

        # Add pH range categories
        df["pH Range"] = pd.cut(
            df["True pH"], bins=[0, 6, 8, 14], labels=["Acidic", "Neutral", "Basic"]
        )

        # Create box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="pH Range", y="Error")
        plt.title(f"Prediction Error by pH Range (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"error_by_range_epoch_{epoch}.png"))
        plt.close()

    def _plot_loss_curves(self, epoch: int):
        """Plot training loss and gradient norm curves."""
        if not self.train_losses:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot training loss
        ax1.plot(self.train_losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel(f"Steps (x{self.log_every_n_steps})")
        ax1.set_ylabel("Loss")

        # Plot gradient norms if available
        if self.gradient_norm_logging and self.grad_norms:
            ax2.plot(self.grad_norms)
            ax2.set_title("Gradient Norm")
            ax2.set_xlabel(f"Steps (x{self.log_every_n_steps})")
            ax2.set_ylabel("Norm")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"training_curves_epoch_{epoch}.png"))
        plt.close()
