import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from Bio import SeqIO
from Bio.PDB import *
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter


def analyze_ph_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Analyze and visualize the distribution of pH values in the dataset."""
    plt.figure(figsize=(12, 6))

    # Create main pH distribution plot
    sns.histplot(data=df, x="pHopt", bins=30, kde=True)
    plt.axvline(x=7.0, color="r", linestyle="--", label="Neutral pH")
    plt.xlabel("pH")
    plt.ylabel("Count")
    plt.title("Distribution of pH Values")
    plt.legend()

    # Add summary statistics
    stats = {
        "Mean": df["pHopt"].mean(),
        "Median": df["pHopt"].median(),
        "Std": df["pHopt"].std(),
        "Min": df["pHopt"].min(),
        "Max": df["pHopt"].max(),
    }

    stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "ph_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def analyze_protein_properties(df: pd.DataFrame, save_dir: str) -> None:
    """Analyze relationships between protein properties and pH."""
    # Calculate protein properties
    properties = []
    for seq in df["Sequence"].unique():
        try:
            analysis = ProteinAnalysis(seq)
            properties.append(
                {
                    "sequence": seq,
                    "length": len(seq),
                    "molecular_weight": analysis.molecular_weight(),
                    "aromaticity": analysis.aromaticity(),
                    "instability_index": analysis.instability_index(),
                    "isoelectric_point": analysis.isoelectric_point(),
                    "charge_at_ph7": analysis.charge_at_pH(7.0),
                }
            )
        except Exception as e:
            print(f"Error analyzing sequence: {e}")
            continue

    prop_df = pd.DataFrame(properties)

    # Merge with original dataframe
    merged_df = df.merge(prop_df, left_on="Sequence", right_on="sequence")

    # Create correlation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Protein Properties vs. pH", fontsize=16)

    # Plot length vs pH
    ax = axes[0, 0]
    sns.scatterplot(data=merged_df, x="length", y="pHopt", alpha=0.5, ax=ax)
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("pH")

    # Plot isoelectric point vs pH
    ax = axes[0, 1]
    sns.scatterplot(data=merged_df, x="isoelectric_point", y="pHopt", alpha=0.5, ax=ax)
    ax.set_xlabel("Isoelectric Point")
    ax.set_ylabel("pH")

    # Plot charge at pH 7 vs pH
    ax = axes[1, 0]
    sns.scatterplot(data=merged_df, x="charge_at_ph7", y="pHopt", alpha=0.5, ax=ax)
    ax.set_xlabel("Charge at pH 7")
    ax.set_ylabel("pH")

    # Plot instability index vs pH
    ax = axes[1, 1]
    sns.scatterplot(data=merged_df, x="instability_index", y="pHopt", alpha=0.5, ax=ax)
    ax.set_xlabel("Instability Index")
    ax.set_ylabel("pH")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "protein_properties.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Calculate and save correlations
    correlations = merged_df[
        ["pHopt", "length", "isoelectric_point", "charge_at_ph7", "instability_index"]
    ].corr()
    correlations.to_csv(os.path.join(save_dir, "property_correlations.csv"))


def create_sequence_similarity_network(
    df: pd.DataFrame, save_dir: str, threshold: float = 0.3
) -> None:
    """Create and visualize sequence similarity network."""
    # Calculate pairwise sequence similarities
    sequences = df["Sequence"].unique()
    n_seq = len(sequences)

    # Create distance matrix
    distances = np.zeros((n_seq, n_seq))
    for i in range(n_seq):
        for j in range(i + 1, n_seq):
            # Calculate sequence identity
            matches = sum(a == b for a, b in zip(sequences[i], sequences[j]))
            similarity = matches / min(len(sequences[i]), len(sequences[j]))
            distances[i, j] = distances[j, i] = 1 - similarity

    # Create network
    G = nx.Graph()
    for i in range(n_seq):
        G.add_node(i, sequence=sequences[i])
        for j in range(i + 1, n_seq):
            if distances[i, j] <= 1 - threshold:
                G.add_edge(i, j, weight=1 - distances[i, j])

    # Calculate node properties
    degrees = dict(G.degree())

    # Create interactive visualization using plotly
    pos = nx.spring_layout(G, k=1 / np.sqrt(n_seq))

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
    )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                size=10,
                color=list(degrees.values()),
                colorbar=dict(title="Node Degree"),
            ),
            text=[f"Sequence {i}<br>Degree: {degrees[i]}" for i in G.nodes()],
        )
    )

    fig.update_layout(
        title="Sequence Similarity Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
    )

    fig.write_html(os.path.join(save_dir, "sequence_similarity_network.html"))


def analyze_experimental_conditions(df: pd.DataFrame, save_dir: str) -> None:
    """Analyze experimental conditions in the dataset."""
    # Create figure for experimental conditions
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Experimental Conditions Analysis", fontsize=16)

    # Temperature distribution
    ax = axes[0, 0]
    if "temperature" in df.columns:
        sns.histplot(data=df, x="temperature", bins=30, ax=ax)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Count")
        ax.set_title("Temperature Distribution")
    else:
        ax.text(0.5, 0.5, "No temperature data available", ha="center", va="center")

    # Method distribution
    ax = axes[0, 1]
    if "method" in df.columns:
        method_counts = df["method"].value_counts()
        method_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Method")
        ax.set_ylabel("Count")
        ax.set_title("Measurement Methods")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No method data available", ha="center", va="center")

    # Organism distribution
    ax = axes[1, 0]
    if "Organism" in df.columns:
        org_counts = df["Organism"].value_counts().head(10)
        org_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Organism")
        ax.set_ylabel("Count")
        ax.set_title("Top 10 Organisms")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No organism data available", ha="center", va="center")

    # EC number distribution
    ax = axes[1, 1]
    if "EC_number" in df.columns:
        ec_counts = df["EC_number"].value_counts().head(10)
        ec_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("EC Number")
        ax.set_ylabel("Count")
        ax.set_title("Top 10 EC Numbers")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No EC number data available", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "experimental_conditions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(dataset_file: str, save_dir: str) -> None:
    """Run comprehensive analysis of the pH dataset."""
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(dataset_file)

    # Analyze pH distribution
    analyze_ph_distribution(df, save_dir)

    # Analyze protein properties
    analyze_protein_properties(df, save_dir)

    # Create sequence similarity network
    create_sequence_similarity_network(df, save_dir)

    # Analyze experimental conditions
    analyze_experimental_conditions(df, save_dir)

    # Save summary statistics
    summary = {
        "total_entries": len(df),
        "unique_proteins": len(df["Sequence"].unique()),
        "unique_organisms": (
            len(df["Organism"].unique()) if "Organism" in df.columns else 0
        ),
        "unique_ec_numbers": (
            len(df["EC_number"].unique()) if "EC_number" in df.columns else 0
        ),
        "ph_range": [df["pHopt"].min(), df["pHopt"].max()],
        "mean_ph": df["pHopt"].mean(),
        "median_ph": df["pHopt"].median(),
    }

    with open(os.path.join(save_dir, "dataset_summary.txt"), "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pH dataset characteristics")
    parser.add_argument("dataset_file", help="Path to dataset CSV file")
    parser.add_argument(
        "--save_dir", default="dataset_analysis", help="Directory to save results"
    )

    args = parser.parse_args()
    main(args.dataset_file, args.save_dir)
