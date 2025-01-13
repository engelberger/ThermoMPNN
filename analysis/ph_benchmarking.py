import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import *
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets import PHDataset
from transfer_model_ph import TransferModelPH
from train_thermompnn_ph import TransferModelPHPL
from protein_mpnn_utils import tied_featurize


def compute_centrality(
    xyz, basis_atom: str = "CA", radius: float = 10.0, chain: str = "A"
) -> torch.Tensor:
    """Compute residue centrality based on neighbor count."""
    coords = xyz[basis_atom + f"_chain_{chain}"]
    coords = torch.tensor(coords)
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1
    return num_neighbors


def get_metrics(device="cuda"):
    """Get metrics for pH prediction evaluation."""
    return {
        "r2": R2Score().to(device),
        "mse": MeanSquaredError(squared=True).to(device),
        "rmse": MeanSquaredError(squared=False).to(device),
        "spearman": SpearmanCorrCoef().to(device),
        "pearson": PearsonCorrCoef().to(device),
    }


def get_trained_model(
    model_path: str, config: dict, checkpt_dir: str = "models/"
) -> nn.Module:
    """Load a trained pH prediction model."""
    if os.path.exists(model_path):
        return TransferModelPHPL.load_from_checkpoint(model_path, cfg=config).model
    else:
        # Try relative to the project root
        model_loc = os.path.join(parent_dir, model_path)
        if os.path.exists(model_loc):
            return TransferModelPHPL.load_from_checkpoint(model_loc, cfg=config).model
        else:
            # Try in the checkpoints directory
            model_loc = os.path.join(parent_dir, checkpt_dir, model_path)
            return TransferModelPHPL.load_from_checkpoint(model_loc, cfg=config).model


def run_prediction_default(
    name: str, model: nn.Module, dataset_name: str, dataset: PHDataset, results: list
) -> list:
    """Run standard inference on a dataset."""
    metrics = {"pH": get_metrics()}
    print(f"Testing Model {name} on dataset {dataset_name}")

    for batch in tqdm(dataset):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        for mut, out in zip(mutations, pred):
            if mut.pH is not None:
                mut.pH = mut.pH.to("cuda")
                for metric in metrics["pH"].values():
                    metric.update(out["pH"], mut.pH)

    column = {
        "Model": name,
        "Dataset": dataset_name,
    }

    for met_name, metric in metrics["pH"].items():
        try:
            value = metric.compute().cpu().item()
            column[f"pH_{met_name}"] = value
            print(f"{met_name}: {value:.3f}")
        except ValueError:
            pass

    results.append(column)
    return results


def run_prediction_with_analysis(
    name: str,
    model: nn.Module,
    dataset_name: str,
    dataset: PHDataset,
    results: list,
    save_dir: str,
    analyze_structure: bool = True,
) -> list:
    """Run inference with detailed analysis."""
    metrics = {"pH": get_metrics()}
    print(f"\nRunning detailed analysis for {name} on {dataset_name}")

    # Initialize results dataframe
    raw_pred_df = pd.DataFrame(
        columns=[
            "Model",
            "Dataset",
            "pH_true",
            "pH_pred",
            "position",
            "wildtype",
            "mutation",
            "pdb_id",
            "EC_number",
            "neighbors",
            "error",
        ]
    )
    row = 0

    total_batches = len(dataset)
    print(f"Processing {total_batches} proteins...")

    for batch_idx, batch in enumerate(tqdm(dataset, desc="Processing proteins")):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        if analyze_structure:
            if batch_idx == 0:
                print("Computing structural properties...")
            coord_chain = [c for c in pdb[0].keys() if "coords" in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(pdb[0][coord_chain], chain=chain)

        for mut, out in zip(mutations, pred):
            if mut.pH is not None:
                mut.pH = mut.pH.to("cuda")
                for metric in metrics["pH"].values():
                    metric.update(out["pH"], mut.pH)

                # Record prediction details
                raw_pred_df.loc[row] = {
                    "Model": name,
                    "Dataset": dataset_name,
                    "pH_true": mut.pH.cpu().item(),
                    "pH_pred": out["pH"].cpu().item(),
                    "position": mut.position,
                    "wildtype": mut.wildtype,
                    "mutation": mut.mutation,
                    "pdb_id": mut.pdb.strip(".pdb"),
                    "EC_number": (
                        dataset.get_ec_number(mut.pdb)
                        if hasattr(dataset, "get_ec_number")
                        else None
                    ),
                    "neighbors": (
                        neighbors[mut.position].cpu().item()
                        if analyze_structure
                        else None
                    ),
                    "error": abs(out["pH"].cpu().item() - mut.pH.cpu().item()),
                }
                row += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{total_batches} proteins")

    # Calculate overall metrics
    print("\nCalculating final metrics...")
    column = {"Model": name, "Dataset": dataset_name}
    for met_name, metric in metrics["pH"].items():
        try:
            value = metric.compute().cpu().item()
            column[f"pH_{met_name}"] = value
            print(f"{met_name}: {value:.3f}")
        except ValueError:
            pass
    results.append(column)

    # Save raw predictions
    print("\nSaving predictions and generating plots...")
    pred_path = os.path.join(save_dir, f"{name}_{dataset_name}_predictions.csv")
    raw_pred_df.to_csv(pred_path)
    print(f"Predictions saved to: {pred_path}")

    # Generate analysis plots
    print("Creating analysis plots...")
    create_analysis_plots(raw_pred_df, name, dataset_name, save_dir)
    print("Analysis plots created successfully")

    return results


def analyze_ph_ranges_detailed(
    df: pd.DataFrame, save_dir: str, bin_width: float = 0.5
) -> dict:
    """Analyze model performance across fine-grained pH ranges."""
    results = {}

    # Create fine-grained pH bins
    bins = np.arange(0, 14 + bin_width, bin_width)
    df["ph_bin"] = pd.cut(df["pH_true"], bins=bins, labels=bins[:-1])

    # Initialize results storage
    bin_stats = []

    # Calculate statistics for each bin
    for bin_start in bins[:-1]:
        bin_data = df[df["ph_bin"] == bin_start]
        if len(bin_data) > 0:
            rmse = np.sqrt(mean_squared_error(bin_data["pH_true"], bin_data["pH_pred"]))
            mae = np.mean(np.abs(bin_data["pH_true"] - bin_data["pH_pred"]))
            r2 = (
                r2_score(bin_data["pH_true"], bin_data["pH_pred"])
                if len(bin_data) > 1
                else np.nan
            )
            spearman = (
                stats.spearmanr(bin_data["pH_true"], bin_data["pH_pred"])[0]
                if len(bin_data) > 1
                else np.nan
            )

            bin_stats.append(
                {
                    "bin_start": bin_start,
                    "bin_end": bin_start + bin_width,
                    "count": len(bin_data),
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "spearman": spearman,
                    "mean_error": np.mean(bin_data["pH_pred"] - bin_data["pH_true"]),
                    "std_error": np.std(bin_data["pH_pred"] - bin_data["pH_true"]),
                }
            )

    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame(bin_stats)

    # Create publication-quality visualization
    plt.style.use("seaborn-whitegrid")
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 1. Sample distribution and RMSE
    ax1 = fig.add_subplot(gs[0, :])
    twin_ax = ax1.twinx()

    # Plot sample counts
    bars = twin_ax.bar(
        stats_df["bin_start"],
        stats_df["count"],
        width=bin_width,
        alpha=0.3,
        color="gray",
        label="Sample count",
    )
    twin_ax.set_ylabel("Number of samples", color="gray")
    twin_ax.tick_params(axis="y", labelcolor="gray")

    # Plot RMSE
    line = ax1.plot(
        stats_df["bin_start"], stats_df["rmse"], color="red", marker="o", label="RMSE"
    )
    ax1.set_ylabel("RMSE", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = twin_ax.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    ax1.set_title(
        "Performance and Sample Distribution Across pH Range",
        pad=20,
        fontsize=12,
        fontweight="bold",
    )

    # 2. Error distribution heatmap
    ax2 = fig.add_subplot(gs[1, :])

    # Create 2D histogram of true vs predicted pH
    hist2d = ax2.hist2d(
        df["pH_true"],
        df["pH_pred"],
        bins=50,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(hist2d[3], ax=ax2, label="Count (log scale)")

    # Add diagonal line
    ax2.plot([0, 14], [0, 14], "r--", alpha=0.8, label="Perfect prediction")

    ax2.set_xlabel("True pH")
    ax2.set_ylabel("Predicted pH")
    ax2.set_title("Prediction Distribution Heatmap", pad=20)

    # 3. Performance metrics
    ax3 = fig.add_subplot(gs[2, 0])
    metrics_to_plot = ["r2", "spearman", "mae"]
    colors = ["blue", "green", "orange"]

    for metric, color in zip(metrics_to_plot, colors):
        ax3.plot(
            stats_df["bin_start"],
            stats_df[metric],
            marker="o",
            label=metric.upper(),
            color=color,
        )

    ax3.set_xlabel("pH")
    ax3.set_ylabel("Metric Value")
    ax3.legend()
    ax3.set_title("Performance Metrics by pH Range")

    # 4. Bias analysis
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.errorbar(
        stats_df["bin_start"],
        stats_df["mean_error"],
        yerr=stats_df["std_error"],
        fmt="o",
        capsize=5,
        label="Mean error ± std",
    )
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax4.set_xlabel("pH")
    ax4.set_ylabel("Prediction Bias")
    ax4.legend()
    ax4.set_title("Prediction Bias Analysis")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "detailed_ph_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save detailed statistics
    stats_df.to_csv(os.path.join(save_dir, "ph_range_statistics.csv"))

    return stats_df.to_dict("records")


def analyze_classification_metrics(df: pd.DataFrame, save_dir: str):
    """Analyze classification metrics for different pH thresholds."""
    # Create multiple pH thresholds for binary classification
    thresholds = [5, 6, 7, 8, 9]
    results = {}

    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)

    # 1. ROC curves for different thresholds
    ax1 = plt.subplot(gs[0, 0])

    for threshold in thresholds:
        # Create binary labels
        y_true = (df["pH_true"] > threshold).astype(int)
        y_pred = (df["pH_pred"] > threshold).astype(int)
        y_score = df["pH_pred"]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax1.plot(fpr, tpr, label=f"pH {threshold} (AUC = {roc_auc:.2f})")

        # Calculate additional metrics
        results[f"threshold_{threshold}"] = {
            "auc": roc_auc,
            "f1": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves for Different pH Thresholds")
    ax1.legend()

    # 2. Precision-Recall curves
    ax2 = plt.subplot(gs[0, 1])

    for threshold in thresholds:
        y_true = (df["pH_true"] > threshold).astype(int)
        y_score = df["pH_pred"]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        ax2.plot(recall, precision, label=f"pH {threshold} (AP = {avg_precision:.2f})")

        results[f"threshold_{threshold}"]["average_precision"] = avg_precision

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend()

    # 3. F1 Scores across pH range
    ax3 = plt.subplot(gs[1, 0])

    ph_range = np.arange(4, 11, 0.5)
    f1_scores = []

    for ph in ph_range:
        y_true = (df["pH_true"] > ph).astype(int)
        y_pred = (df["pH_pred"] > ph).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    ax3.plot(ph_range, f1_scores, marker="o")
    ax3.set_xlabel("pH Threshold")
    ax3.set_ylabel("F1 Score")
    ax3.set_title("F1 Scores Across pH Range")

    # 4. Confusion Matrix Heatmap for pH 7
    ax4 = plt.subplot(gs[1, 1])

    y_true = (df["pH_true"] > 7).astype(int)
    y_pred = (df["pH_pred"] > 7).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax4)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("True")
    ax4.set_title("Normalized Confusion Matrix (pH 7)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "classification_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save results
    pd.DataFrame(results).to_csv(os.path.join(save_dir, "classification_metrics.csv"))

    return results


def create_analysis_plots(raw_pred_df, name, dataset_name, save_dir):
    """Create comprehensive analysis plots with enhanced aesthetics."""
    # Set modern style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Create figure with GridSpec for better layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Scatter plot with density
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=raw_pred_df, x="true_pH", y="pred_pH", alpha=0.5, ax=ax1)
    ax1.plot([0, 14], [0, 14], "r--", alpha=0.7)
    ax1.set_xlabel("Experimental pH")
    ax1.set_ylabel("Predicted pH")
    ax1.set_title(
        f"Prediction vs. Experimental pH\nR² = {r2_score(raw_pred_df.true_pH, raw_pred_df.pred_pH):.3f}"
    )

    # Error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    errors = raw_pred_df.pred_pH - raw_pred_df.true_pH
    sns.histplot(data=errors, bins=30, kde=True, ax=ax2)
    ax2.set_xlabel("Prediction Error (pH units)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")

    # pH range analysis
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.arange(0, 14.5, 0.5)
    raw_pred_df["pH_bin"] = pd.cut(raw_pred_df.true_pH, bins)
    bin_stats = (
        raw_pred_df.groupby("pH_bin")
        .agg(
            {
                "pred_pH": lambda x: np.sqrt(
                    mean_squared_error([y for y in x], [x.mean() for _ in x])
                )
            }
        )
        .reset_index()
    )
    bin_counts = raw_pred_df.groupby("pH_bin").size()

    sns.barplot(data=bin_stats, x="pH_bin", y="pred_pH", ax=ax3)
    ax3.set_xlabel("pH Range")
    ax3.set_ylabel("RMSE")
    ax3.set_title("Performance by pH Range")
    ax3.tick_params(axis="x", rotation=45)

    # Add sample counts as text
    for i, count in enumerate(bin_counts):
        ax3.text(i, 0.1, f"n={count}", ha="center", va="bottom", rotation=90)

    # Residual plot
    ax4 = fig.add_subplot(gs[1, 1])
    sns.regplot(
        data=raw_pred_df,
        x="true_pH",
        y=errors,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ax=ax4,
    )
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Experimental pH")
    ax4.set_ylabel("Prediction Error")
    ax4.set_title("Residual Analysis")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{name}_{dataset_name}_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create additional plots for detailed analysis
    create_detailed_analysis_plots(raw_pred_df, name, dataset_name, save_dir)


def create_detailed_analysis_plots(raw_pred_df, name, dataset_name, save_dir):
    """Create additional detailed analysis plots."""
    # Set style
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Performance metrics by pH range with confidence intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # RMSE by pH range with error bars
    bins = np.arange(0, 14.5, 1.0)
    raw_pred_df["pH_bin"] = pd.cut(raw_pred_df.true_pH, bins)

    def bootstrap_rmse(group):
        n_bootstrap = 1000
        rmses = []
        for _ in range(n_bootstrap):
            sample = group.sample(n=len(group), replace=True)
            rmse = np.sqrt(mean_squared_error(sample.true_pH, sample.pred_pH))
            rmses.append(rmse)
        return pd.Series(
            {
                "rmse": np.mean(rmses),
                "rmse_ci_low": np.percentile(rmses, 2.5),
                "rmse_ci_high": np.percentile(rmses, 97.5),
            }
        )

    bin_stats = raw_pred_df.groupby("pH_bin").apply(bootstrap_rmse).reset_index()

    sns.barplot(data=bin_stats, x="pH_bin", y="rmse", ax=ax1)
    ax1.errorbar(
        x=range(len(bin_stats)),
        y=bin_stats.rmse,
        yerr=[bin_stats.rmse_ci_low, bin_stats.rmse_ci_high - bin_stats.rmse],
        fmt="none",
        color="black",
        capsize=5,
    )
    ax1.set_xlabel("pH Range")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE by pH Range with 95% CI")
    ax1.tick_params(axis="x", rotation=45)

    # Sample density plot
    sns.kdeplot(data=raw_pred_df, x="true_pH", y="pred_pH", fill=True, ax=ax2)
    ax2.plot([0, 14], [0, 14], "r--", alpha=0.7)
    ax2.set_xlabel("Experimental pH")
    ax2.set_ylabel("Predicted pH")
    ax2.set_title("Prediction Density Plot")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{name}_{dataset_name}_detailed_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def bootstrap_confidence_intervals(
    df: pd.DataFrame, n_iterations: int = 1000, confidence_level: float = 0.95
) -> pd.DataFrame:
    """Calculate confidence intervals using bootstrapping."""
    results = []
    for _ in range(n_iterations):
        sample = df.sample(n=len(df), replace=True)
        results.append(
            {
                "rmse": np.sqrt(((sample["pH_pred"] - sample["pH_true"]) ** 2).mean()),
                "r2": stats.pearsonr(sample["pH_true"], sample["pH_pred"])[0] ** 2,
                "spearman": stats.spearmanr(sample["pH_true"], sample["pH_pred"])[0],
            }
        )

    results_df = pd.DataFrame(results)
    alpha = (1 - confidence_level) / 2
    ci = {}
    for metric in results_df.columns:
        ci[metric] = {
            "mean": results_df[metric].mean(),
            "lower": results_df[metric].quantile(alpha),
            "upper": results_df[metric].quantile(1 - alpha),
        }

    return pd.DataFrame(ci).round(3)


def create_structural_analysis_plot(
    df: pd.DataFrame, model_name: str, dataset_name: str, save_dir: str
):
    """Create enhanced structural analysis plots."""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)

    # 1. Error vs neighbors with density estimation
    ax1 = plt.subplot(gs[0, 0])

    # Calculate point density
    xy = np.vstack([df["neighbors"], np.abs(df["pH_pred"] - df["pH_true"])])
    z = gaussian_kde(xy)(xy)

    # Sort points by density
    idx = z.argsort()
    x, y, z = (
        df["neighbors"].values[idx],
        np.abs(df["pH_pred"].values[idx] - df["pH_true"].values[idx]),
        z[idx],
    )

    scatter = ax1.scatter(x, y, c=z, s=50, alpha=0.5, cmap="viridis")
    plt.colorbar(scatter, label="Density")

    # Add trend line
    z = np.polyfit(df["neighbors"], np.abs(df["pH_pred"] - df["pH_true"]), 1)
    p = np.poly1d(z)
    ax1.plot(df["neighbors"], p(df["neighbors"]), "r--", alpha=0.8)

    # Calculate correlation
    corr, pval = stats.spearmanr(df["neighbors"], np.abs(df["pH_pred"] - df["pH_true"]))

    ax1.text(
        0.05,
        0.95,
        f"Spearman ρ = {corr:.3f}\np-value = {pval:.2e}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("Number of Neighbors")
    ax1.set_ylabel("Absolute Error")
    ax1.set_title("Error vs. Residue Centrality")

    # 2. Error distribution by neighbor count bins
    ax2 = plt.subplot(gs[0, 1])

    # Create neighbor count bins
    n_bins = 5
    df["neighbor_bin"] = pd.qcut(
        df["neighbors"], n_bins, labels=[f"Q{i+1}" for i in range(n_bins)]
    )

    sns.boxplot(
        data=df, x="neighbor_bin", y=np.abs(df["pH_pred"] - df["pH_true"]), ax=ax2
    )
    ax2.set_xlabel("Neighbor Count Quintile")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Error Distribution by Neighbor Count")

    # Add sample sizes
    for i, bin_name in enumerate(df["neighbor_bin"].unique()):
        n = len(df[df["neighbor_bin"] == bin_name])
        ax2.text(
            i,
            ax2.get_ylim()[1],
            f"n={n}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    # 3. Performance metrics by neighbor count
    ax3 = plt.subplot(gs[1, :])

    metrics = []
    for bin_name in df["neighbor_bin"].unique():
        bin_data = df[df["neighbor_bin"] == bin_name]
        metrics.append(
            {
                "bin": bin_name,
                "rmse": np.sqrt(
                    mean_squared_error(bin_data["pH_true"], bin_data["pH_pred"])
                ),
                "r2": r2_score(bin_data["pH_true"], bin_data["pH_pred"]),
                "spearman": stats.spearmanr(bin_data["pH_true"], bin_data["pH_pred"])[
                    0
                ],
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = pd.melt(
        metrics_df, id_vars=["bin"], var_name="metric", value_name="value"
    )

    sns.barplot(data=metrics_df, x="bin", y="value", hue="metric", ax=ax3)
    ax3.set_xlabel("Neighbor Count Quintile")
    ax3.set_ylabel("Metric Value")
    ax3.set_title("Performance Metrics by Neighbor Count")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_structural_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save structural analysis statistics
    metrics_df.to_csv(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_structural_metrics.csv")
    )


def create_ec_analysis_plot(
    df: pd.DataFrame, model_name: str, dataset_name: str, save_dir: str
):
    """Create enhanced EC number analysis plots."""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2)

    # 1. Error distribution by EC number
    ax1 = plt.subplot(gs[0, :])

    # Calculate statistics
    ec_stats = (
        df.groupby("EC_number")
        .agg(
            {
                "error": ["mean", "std", "count", "median"],
                "pH_true": ["mean", "std"],
                "pH_pred": ["mean", "std"],
            }
        )
        .round(3)
    )

    # Sort EC numbers by mean error
    ec_order = ec_stats["error"]["mean"].sort_values().index

    # Create violin plot
    sns.violinplot(data=df, x="EC_number", y="error", order=ec_order, ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # Add sample sizes
    for i, ec in enumerate(ec_order):
        n = ec_stats["error"]["count"][ec]
        ax1.text(
            i,
            ax1.get_ylim()[1],
            f"n={n}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    ax1.set_title("Error Distribution by EC Number")

    # 2. Performance metrics by EC number
    ax2 = plt.subplot(gs[1, 0])
    metrics = []

    for ec in df["EC_number"].unique():
        ec_data = df[df["EC_number"] == ec]
        metrics.append(
            {
                "EC": ec,
                "RMSE": np.sqrt(
                    mean_squared_error(ec_data["pH_true"], ec_data["pH_pred"])
                ),
                "R²": r2_score(ec_data["pH_true"], ec_data["pH_pred"]),
                "Spearman": stats.spearmanr(ec_data["pH_true"], ec_data["pH_pred"])[0],
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = pd.melt(
        metrics_df, id_vars=["EC"], var_name="Metric", value_name="Value"
    )

    sns.barplot(data=metrics_df, x="EC", y="Value", hue="Metric", ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_title("Performance Metrics by EC Number")

    # 3. pH distribution by EC number
    ax3 = plt.subplot(gs[1, 1])

    sns.boxplot(data=df, x="EC_number", y="pH_true", order=ec_order, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_title("pH Distribution by EC Number")

    # 4. Statistical significance analysis
    ax4 = plt.subplot(gs[2, :])

    # Perform pairwise statistical tests
    ec_pairs = []
    for i, ec1 in enumerate(df["EC_number"].unique()):
        for ec2 in df["EC_number"].unique()[i + 1 :]:
            errors1 = df[df["EC_number"] == ec1]["error"]
            errors2 = df[df["EC_number"] == ec2]["error"]

            stat, pval = stats.mannwhitneyu(errors1, errors2)
            effect_size = compute_effect_size(errors1.values, errors2.values)

            ec_pairs.append(
                {"EC1": ec1, "EC2": ec2, "p_value": pval, "effect_size": effect_size}
            )

    # Create significance matrix
    ec_numbers = sorted(df["EC_number"].unique())
    n_ec = len(ec_numbers)
    sig_matrix = np.zeros((n_ec, n_ec))
    effect_matrix = np.zeros((n_ec, n_ec))

    for pair in ec_pairs:
        i = ec_numbers.index(pair["EC1"])
        j = ec_numbers.index(pair["EC2"])
        sig_matrix[i, j] = -np.log10(pair["p_value"])
        sig_matrix[j, i] = -np.log10(pair["p_value"])
        effect_matrix[i, j] = pair["effect_size"]
        effect_matrix[j, i] = -pair["effect_size"]

    # Plot significance heatmap
    im = ax4.imshow(sig_matrix, cmap="YlOrRd")
    plt.colorbar(im, ax=ax4, label="-log10(p-value)")

    # Add effect size annotations
    for i in range(n_ec):
        for j in range(n_ec):
            if i != j:
                ax4.text(
                    j,
                    i,
                    f"{effect_matrix[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if sig_matrix[i, j] > 2 else "black",
                )

    ax4.set_xticks(range(n_ec))
    ax4.set_yticks(range(n_ec))
    ax4.set_xticklabels(ec_numbers, rotation=45, ha="right")
    ax4.set_yticklabels(ec_numbers)
    ax4.set_title("Statistical Significance and Effect Size Matrix")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_ec_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save EC analysis statistics
    ec_stats.to_csv(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_ec_statistics.csv")
    )
    pd.DataFrame(ec_pairs).to_csv(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_ec_pairwise_tests.csv")
    )


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (np.mean(group1) - np.mean(group2)) / pooled_se


def main(cfg, args):
    """Run benchmarking."""
    print("\n=== Starting pH Prediction Benchmarking ===")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")

    # Initialize models
    print("\nLoading models...")
    models = {
        "ThermoMPNN-pH": get_trained_model(model_path=args.model_path, config=cfg)
    }
    print("Models loaded successfully")

    # Initialize datasets
    print("\nLoading datasets...")
    datasets = {}
    for split in ["train", "val", "test"]:
        print(f"Loading {split} dataset...")
        dataset = PHDataset(cfg, split)
        print(f"{split.capitalize()} dataset size: {len(dataset)} proteins")
        datasets[f"pH-{split}"] = dataset
    print("All datasets loaded successfully")

    # Run benchmarking
    results = []
    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        model.eval()
        for dataset_name, dataset in datasets.items():
            print(f"\nBenchmarking on {dataset_name}...")
            if args.detailed_analysis:
                print("Running detailed analysis with plots...")
                results = run_prediction_with_analysis(
                    name,
                    model,
                    dataset_name,
                    dataset,
                    results,
                    args.save_dir,
                    args.analyze_structure,
                )
            else:
                results = run_prediction_default(
                    name, model, dataset_name, dataset, results
                )

    # Save results
    print("\nSaving benchmark results...")
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.save_dir, "benchmark_results.csv")
    results_df.to_csv(results_path)
    print(f"Results saved to: {results_path}")

    # Generate confidence intervals if detailed analysis
    if args.detailed_analysis:
        print("\nCalculating confidence intervals...")
        for name, model in models.items():
            for dataset_name, _ in datasets.items():
                print(f"Processing {name} on {dataset_name}...")
                pred_file = os.path.join(
                    args.save_dir, f"{name}_{dataset_name}_predictions.csv"
                )
                if os.path.exists(pred_file):
                    print(f"Loading predictions from: {pred_file}")
                    pred_df = pd.read_csv(pred_file)
                    print("Computing bootstrap confidence intervals...")
                    ci_df = bootstrap_confidence_intervals(pred_df)
                    ci_path = os.path.join(
                        args.save_dir,
                        f"{name}_{dataset_name}_confidence_intervals.csv",
                    )
                    ci_df.to_csv(ci_path)
                    print(f"Confidence intervals saved to: {ci_path}")

    print("\n=== Benchmarking Complete ===")
    print("\nFinal Results Summary:")
    print(results_df.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark pH prediction models")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument(
        "--save_dir", default="benchmark_results", help="Directory to save results"
    )
    parser.add_argument(
        "--detailed_analysis",
        action="store_true",
        help="Run detailed analysis with plots",
    )
    parser.add_argument(
        "--analyze_structure", action="store_true", help="Include structural analysis"
    )

    args = parser.parse_args()

    # Load the pH prediction config
    config_path = os.path.join(parent_dir, "configs/ph_prediction.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    cfg = OmegaConf.load(config_path)

    with torch.no_grad():
        main(cfg, args)
