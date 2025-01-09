import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import torch
from Bio.PDB import *
from Bio import SeqIO
import plotly.graph_objects as go
import plotly.express as px


def load_predictions(predictions_file: str) -> pd.DataFrame:
    """Load and validate prediction results."""
    df = pd.read_csv(predictions_file)
    required_cols = [
        "pdb_id",
        "position",
        "wildtype",
        "mutation",
        "pH_pred",
        "pH_true",
        "EC_number",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return df


def calculate_metrics_by_group(
    df: pd.DataFrame, group_column: str, metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate performance metrics for each group."""
    if metrics is None:
        metrics = ["rmse", "mae", "r2", "spearman"]

    results = []
    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]

        metric_values = {}
        pred = group_df["pH_pred"].values
        true = group_df["pH_true"].values

        if "rmse" in metrics:
            metric_values["rmse"] = np.sqrt(mean_squared_error(true, pred))
        if "mae" in metrics:
            metric_values["mae"] = np.mean(np.abs(true - pred))
        if "r2" in metrics:
            metric_values["r2"] = r2_score(true, pred)
        if "spearman" in metrics:
            metric_values["spearman"] = stats.spearmanr(true, pred)[0]

        results.append({"group": group, "count": len(group_df), **metric_values})

    return pd.DataFrame(results)


def plot_ec_number_performance(df: pd.DataFrame, save_dir: str) -> None:
    """Create visualization of performance by EC number."""
    # Calculate metrics by EC number
    ec_metrics = calculate_metrics_by_group(df, "EC_number")

    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Performance by EC Number", fontsize=16)

    # Plot RMSE
    ax = axes[0, 0]
    sns.barplot(data=ec_metrics, x="group", y="rmse", ax=ax)
    ax.set_xlabel("EC Number")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=45)

    # Plot count
    ax = axes[0, 1]
    sns.barplot(data=ec_metrics, x="group", y="count", ax=ax)
    ax.set_xlabel("EC Number")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    # Plot R²
    ax = axes[1, 0]
    sns.barplot(data=ec_metrics, x="group", y="r2", ax=ax)
    ax.set_xlabel("EC Number")
    ax.set_ylabel("R²")
    ax.tick_params(axis="x", rotation=45)

    # Plot Spearman correlation
    ax = axes[1, 1]
    sns.barplot(data=ec_metrics, x="group", y="spearman", ax=ax)
    ax.set_xlabel("EC Number")
    ax.set_ylabel("Spearman Correlation")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "ec_number_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_ph_ranges(
    df: pd.DataFrame,
    save_dir: str,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Analyze and visualize performance across pH ranges."""
    if ranges is None:
        ranges = {"acidic": (0, 6), "neutral": (6, 8), "basic": (8, 14)}

    # Create range column
    df["ph_range"] = pd.cut(
        df["pH_true"], bins=[r[0] for r in ranges.values()] + [14], labels=ranges.keys()
    )

    # Calculate metrics by range
    range_metrics = calculate_metrics_by_group(df, "ph_range")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Performance by pH Range", fontsize=16)

    # Plot RMSE
    ax = axes[0, 0]
    sns.barplot(data=range_metrics, x="group", y="rmse", ax=ax)
    ax.set_xlabel("pH Range")
    ax.set_ylabel("RMSE")

    # Plot count
    ax = axes[0, 1]
    sns.barplot(data=range_metrics, x="group", y="count", ax=ax)
    ax.set_xlabel("pH Range")
    ax.set_ylabel("Count")

    # Plot error distribution
    ax = axes[1, 0]
    df["error"] = df["pH_pred"] - df["pH_true"]
    sns.boxplot(data=df, x="ph_range", y="error", ax=ax)
    ax.set_xlabel("pH Range")
    ax.set_ylabel("Prediction Error")
    ax.axhline(y=0, color="r", linestyle="--")

    # Plot scatter by range
    ax = axes[1, 1]
    for range_name in ranges.keys():
        range_df = df[df["ph_range"] == range_name]
        ax.scatter(
            range_df["pH_true"], range_df["pH_pred"], alpha=0.5, label=range_name
        )
    ax.plot([0, 14], [0, 14], "r--")
    ax.set_xlabel("Experimental pH")
    ax.set_ylabel("Predicted pH")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "ph_range_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_structure_visualization(
    df: pd.DataFrame, pdb_id: str, pdb_file: str, save_dir: str
) -> None:
    """Create interactive 3D visualization of prediction errors on structure."""
    # Load structure
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)

    # Get predictions for this structure
    struct_df = df[df["pdb_id"] == pdb_id]

    # Calculate error per residue
    struct_df["error"] = np.abs(struct_df["pH_pred"] - struct_df["pH_true"])

    # Create visualization using plotly
    atoms = []
    colors = []
    texts = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] in struct_df["position"].values:
                    res_data = struct_df[struct_df["position"] == residue.id[1]].iloc[0]
                    error = res_data["error"]

                    # Get CA atom coordinates
                    if "CA" in residue:
                        ca_atom = residue["CA"]
                        atoms.append(ca_atom.get_coord())
                        colors.append(error)
                        texts.append(
                            f"Residue: {residue.get_resname()}{residue.id[1]}<br>"
                            f"True pH: {res_data['pH_true']:.2f}<br>"
                            f"Predicted pH: {res_data['pH_pred']:.2f}<br>"
                            f"Error: {error:.2f}"
                        )

    atoms = np.array(atoms)

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=atoms[:, 0],
                y=atoms[:, 1],
                z=atoms[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="Prediction Error"),
                    showscale=True,
                ),
                text=texts,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=f"pH Prediction Errors on Structure ({pdb_id})",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    fig.write_html(os.path.join(save_dir, f"{pdb_id}_structure_viz.html"))


def bootstrap_confidence_intervals(
    df: pd.DataFrame, n_iterations: int = 1000, confidence_level: float = 0.95
) -> pd.DataFrame:
    """Calculate confidence intervals for metrics using bootstrapping."""
    results = []

    for _ in range(n_iterations):
        # Sample with replacement
        sample = df.sample(n=len(df), replace=True)

        # Calculate metrics
        pred = sample["pH_pred"].values
        true = sample["pH_true"].values

        results.append(
            {
                "rmse": np.sqrt(mean_squared_error(true, pred)),
                "r2": r2_score(true, pred),
                "spearman": stats.spearmanr(true, pred)[0],
            }
        )

    results_df = pd.DataFrame(results)

    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2
    ci = {}
    for metric in results_df.columns:
        ci[metric] = {
            "mean": results_df[metric].mean(),
            "lower": results_df[metric].quantile(alpha),
            "upper": results_df[metric].quantile(1 - alpha),
        }

    return pd.DataFrame(ci).round(3)


def main(predictions_file: str, pdb_dir: str, save_dir: str) -> None:
    """Run comprehensive analysis of pH predictions."""
    os.makedirs(save_dir, exist_ok=True)

    # Load predictions
    df = load_predictions(predictions_file)

    # Calculate overall metrics with confidence intervals
    ci_df = bootstrap_confidence_intervals(df)
    ci_df.to_csv(os.path.join(save_dir, "confidence_intervals.csv"))

    # Analyze performance by EC number
    plot_ec_number_performance(df, save_dir)

    # Analyze performance across pH ranges
    analyze_ph_ranges(df, save_dir)

    # Create structure visualizations for each unique protein
    for pdb_id in df["pdb_id"].unique():
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            create_structure_visualization(df, pdb_id, pdb_file, save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pH prediction results")
    parser.add_argument("predictions_file", help="Path to predictions CSV file")
    parser.add_argument("pdb_dir", help="Directory containing PDB files")
    parser.add_argument(
        "--save_dir", default="analysis_results", help="Directory to save results"
    )

    args = parser.parse_args()
    main(args.predictions_file, args.pdb_dir, args.save_dir)
