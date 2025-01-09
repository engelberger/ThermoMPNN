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

import sys

sys.path.append("../")
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
    if os.path.isabs(model_path):
        return TransferModelPHPL.load_from_checkpoint(model_path, cfg=config).model
    else:
        model_loc = os.path.join(
            config.platform.thermompnn_dir, checkpt_dir, model_path
        )
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
    print(f"Running model {name} on dataset {dataset_name}")

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

    for batch in tqdm(dataset):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        # Calculate structural properties if requested
        if analyze_structure:
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

    # Calculate overall metrics
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
    raw_pred_df.to_csv(os.path.join(save_dir, f"{name}_{dataset_name}_predictions.csv"))

    # Generate analysis plots
    create_analysis_plots(raw_pred_df, name, dataset_name, save_dir)

    return results


def create_analysis_plots(
    df: pd.DataFrame, model_name: str, dataset_name: str, save_dir: str
):
    """Create comprehensive analysis plots."""
    # Set up plotting style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    # 1. Overall prediction scatter plot
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(
        data=df,
        x="pH_true",
        y="pH_pred",
        kind="hex",
        joint_kws={"gridsize": 20},
        marginal_kws={"bins": 30},
    )
    g.fig.suptitle(f"{model_name} on {dataset_name}")
    g.ax_joint.plot([0, 14], [0, 14], "r--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_{dataset_name}_predictions.png"))
    plt.close()

    # 2. Error distribution by pH range
    ranges = {"acidic": (0, 6), "neutral": (6, 8), "basic": (8, 14)}
    df["ph_range"] = pd.cut(
        df["pH_true"], bins=[r[0] for r in ranges.values()] + [14], labels=ranges.keys()
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="ph_range", y="error")
    plt.title(f"Prediction Error by pH Range - {model_name}")
    plt.savefig(
        os.path.join(save_dir, f"{model_name}_{dataset_name}_error_by_range.png")
    )
    plt.close()

    # 3. Error vs. structural properties
    if "neighbors" in df.columns and df["neighbors"].notna().any():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="neighbors", y="error", alpha=0.5)
        plt.title(f"Error vs. Residue Centrality - {model_name}")
        plt.savefig(
            os.path.join(
                save_dir, f"{model_name}_{dataset_name}_error_vs_centrality.png"
            )
        )
        plt.close()

    # 4. EC number analysis
    if "EC_number" in df.columns and df["EC_number"].notna().any():
        ec_metrics = (
            df.groupby("EC_number")
            .agg(
                {
                    "error": ["mean", "std", "count"],
                    "pH_true": ["mean", "std"],
                }
            )
            .round(3)
        )
        ec_metrics.to_csv(
            os.path.join(save_dir, f"{model_name}_{dataset_name}_ec_analysis.csv")
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="EC_number", y="error")
        plt.xticks(rotation=45)
        plt.title(f"Error by EC Number - {model_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{model_name}_{dataset_name}_error_by_ec.png")
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


def main(cfg: dict, args: dict):
    """Run comprehensive pH prediction benchmarking."""
    os.makedirs(args.save_dir, exist_ok=True)

    # Load models to benchmark
    models = {
        "ThermoMPNN-pH": get_trained_model(model_path=args.model_path, config=cfg)
    }

    # Load benchmark datasets
    datasets = {
        "pH-test": PHDataset(cfg, "test"),
        "pH-homologue-free": PHDataset(cfg, "homologue-free"),
    }

    results = []
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()

        for dataset_name, dataset in datasets.items():
            if args.detailed_analysis:
                results = run_prediction_with_analysis(
                    name=name,
                    model=model,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    results=results,
                    save_dir=args.save_dir,
                    analyze_structure=args.analyze_structure,
                )
            else:
                results = run_prediction_default(
                    name=name,
                    model=model,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    results=results,
                )

    # Save overall results
    df = pd.DataFrame(results)
    print("\nOverall Results:")
    print(df)
    df.to_csv(os.path.join(args.save_dir, "benchmark_results.csv"))

    # Generate confidence intervals if detailed analysis
    if args.detailed_analysis:
        for name, model in models.items():
            for dataset_name, _ in datasets.items():
                pred_file = os.path.join(
                    args.save_dir, f"{name}_{dataset_name}_predictions.csv"
                )
                if os.path.exists(pred_file):
                    pred_df = pd.read_csv(pred_file)
                    ci_df = bootstrap_confidence_intervals(pred_df)
                    ci_df.to_csv(
                        os.path.join(
                            args.save_dir,
                            f"{name}_{dataset_name}_confidence_intervals.csv",
                        )
                    )


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
    cfg = OmegaConf.load("../local.yaml")

    with torch.no_grad():
        main(cfg, args)
